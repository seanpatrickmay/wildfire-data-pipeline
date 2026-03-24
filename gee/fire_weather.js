// NOTE: This script is for interactive exploration in the GEE Code Editor.
// For batch processing, use the Python API: `wildfire download <fire_name>`

/**
 * Fire Weather Index Export
 *
 * Supplements RTMA raw weather with pre-computed fire behavior indices:
 * 1. GRIDMET ERC (Energy Release Component) — daily, 4km
 * 2. GRIDMET Burning Index — daily, 4km
 * 3. GRIDMET 100-hour dead fuel moisture — daily, 4km
 * 4. GRIDMET 1000-hour dead fuel moisture — daily, 4km
 * 5. GRIDMET VPD (Vapor Pressure Deficit) — daily, 4km
 * 6. ERA5 soil moisture — hourly, 11km
 * 7. GRIDMET PDSI drought index — pentadal, 4km
 *
 * These encode domain-specific fire behavior knowledge that raw
 * temperature/humidity/wind don't directly capture.
 */

// ═══════════════════════════════════════════════════════════════════
// Configuration — can be overridden via URL parameters
// Usage: append ?fire=Walker&year=2019&... to the script URL
// ═══════════════════════════════════════════════════════════════════
var FIRE_NAME = ui.url.get('fire', 'Kincade');
var FIRE_YEAR = parseInt(ui.url.get('year', '2019'), 10);
var aoi_str = ui.url.get('aoi', '-122.96,38.50,-122.59,38.87');
var aoi_parts = aoi_str.split(',').map(Number);
var AOI = ee.Geometry.Rectangle(aoi_parts);
var START_DATE = ee.Date(ui.url.get('start', '2019-10-24'));
var END_DATE = ee.Date(ui.url.get('end', '2019-10-31'));
var EXPORT_SCALE = parseInt(ui.url.get('scale', '2004'), 10);
var EXPORT_CRS = 'EPSG:3857';

// ═══════════════════════════════════════════════════════════════════
// 1. GRIDMET Daily Fire Weather (4km)
// ═══════════════════════════════════════════════════════════════════

var gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(AOI);

// Energy Release Component — proxy for available fire energy
// Higher ERC = more intense fire behavior
var erc = gridmet.select('erc');

// Burning Index — fire spread potential
// Combines spread rate and flame length
var bi = gridmet.select('bi');

// 100-hour dead fuel moisture (%)
// Below 10% = extreme fire danger; 15-20% = moderate
var fm100 = gridmet.select('fm100');

// 1000-hour dead fuel moisture (%)
// Represents large woody fuel readiness
var fm1000 = gridmet.select('fm1000');

// Vapor Pressure Deficit (kPa) — atmospheric drying demand
// Higher VPD = faster fuel drying
var vpd = gridmet.select('vpd');

// Min/max relative humidity
var rmin = gridmet.select('rmin');
var rmax = gridmet.select('rmax');

// Build daily fire weather stack
var dailyFireWeather = gridmet.map(function(img) {
  return img.select(['erc', 'bi', 'fm100', 'fm1000', 'vpd', 'rmin', 'rmax'])
    .set('date', ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'));
});

// Export as multi-band image (one band per day per variable)
var ercStack = erc.toBands().rename(
  erc.aggregate_array('system:time_start').map(function(t) {
    return ee.String('erc_').cat(ee.Date(t).format('YYYYMMdd'));
  })
);

Export.image.toDrive({
  image: ercStack,
  description: FIRE_NAME + '_' + FIRE_YEAR + '_GRIDMET_ERC',
  folder: 'fire_pipeline',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// Export all GRIDMET variables as a single multi-band stack
var allGridmet = dailyFireWeather.toBands();

Export.image.toDrive({
  image: allGridmet,
  description: FIRE_NAME + '_' + FIRE_YEAR + '_GRIDMET_All',
  folder: 'fire_pipeline',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// ═══════════════════════════════════════════════════════════════════
// 2. GRIDMET Drought Indices (pentadal, 4km)
// ═══════════════════════════════════════════════════════════════════

var drought = ee.ImageCollection('GRIDMET/DROUGHT')
  .filterDate(START_DATE.advance(-30, 'day'), END_DATE)
  .filterBounds(AOI);

// Get most recent drought indices before/during fire
var latestDrought = drought.sort('system:time_start', false).first();

var pdsi = latestDrought.select('pdsi').rename('pdsi');
var eddi14 = latestDrought.select('eddi14d').rename('eddi_14d');
var eddi30 = latestDrought.select('eddi30d').rename('eddi_30d');

var droughtStack = pdsi.addBands(eddi14).addBands(eddi30);

Export.image.toDrive({
  image: droughtStack,
  description: FIRE_NAME + '_' + FIRE_YEAR + '_Drought',
  folder: 'fire_pipeline',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// ═══════════════════════════════════════════════════════════════════
// 3. ERA5 Soil Moisture (hourly, ~11km)
// ═══════════════════════════════════════════════════════════════════

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(AOI);

// Volumetric soil water in top layer — dead fuel moisture proxy
var soilMoisture = era5.select('volumetric_soil_water_layer_1');

// Daily mean soil moisture (reduces data volume)
var days = ee.List.sequence(0, END_DATE.difference(START_DATE, 'day').subtract(1));
var dailySoilMoisture = ee.ImageCollection(days.map(function(d) {
  var dayStart = START_DATE.advance(d, 'day');
  var dayEnd = START_DATE.advance(ee.Number(d).add(1), 'day');
  return soilMoisture.filterDate(dayStart, dayEnd).mean()
    .rename('soil_moisture')
    .set('date', dayStart.format('YYYY-MM-dd'));
}));

var smStack = dailySoilMoisture.toBands();

Export.image.toDrive({
  image: smStack,
  description: FIRE_NAME + '_' + FIRE_YEAR + '_SoilMoisture',
  folder: 'fire_pipeline',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

print('GRIDMET collection size:', gridmet.size());
print('ERA5 collection size:', era5.size());
print('Drought bands:', droughtStack.bandNames());
