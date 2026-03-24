// NOTE: This script is for interactive exploration in the GEE Code Editor.
// For batch processing, use the Python API: `wildfire download <fire_name>`

/**
 * Terrain & Fuel Data Export
 *
 * Static spatial features that drive fire spread physics:
 * 1. Slope & aspect from 10m 3DEP DEM (or 30m SRTM fallback)
 * 2. Terrain position (ridge vs valley) for channeling effects
 * 3. Fuel load proxy from NLCD land cover
 * 4. Tree canopy cover
 * 5. Prior burn history from MTBS
 * 6. LANDFIRE vegetation type and height
 *
 * These are exported ONCE per fire AOI (they don't change hourly).
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
var EXPORT_SCALE = parseInt(ui.url.get('scale', '2004'), 10);
var EXPORT_CRS = 'EPSG:3857';

// ═══════════════════════════════════════════════════════════════════
// 1. Terrain (3DEP 10m, resampled to GOES grid)
// ═══════════════════════════════════════════════════════════════════

// Use 3DEP for CONUS (10m lidar), fallback to SRTM (30m) globally
var dem = ee.Image('USGS/3DEP/10m').select('elevation');

// Slope in degrees (fire doubles spread rate per ~10° slope)
var slope = ee.Terrain.slope(dem).rename('slope_deg');

// Aspect as sin/cos (matches wind direction encoding in fire model)
var aspect = ee.Terrain.aspect(dem);
var aspectRad = aspect.multiply(Math.PI / 180);
var aspectSin = aspectRad.sin().rename('aspect_sin');
var aspectCos = aspectRad.cos().rename('aspect_cos');

// Topographic Position Index — ridge (+) vs valley (-)
// Local scale (~500m): captures gullies and small ridges
var meanElevLocal = dem.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: ee.Kernel.circle(500, 'meters')
});
var tpi = dem.subtract(meanElevLocal).rename('tpi');

// Terrain ruggedness (std dev of elevation in neighborhood)
var ruggedness = dem.reduceNeighborhood({
  reducer: ee.Reducer.stdDev(),
  kernel: ee.Kernel.circle(500, 'meters')
}).rename('terrain_ruggedness');

// Elevation itself (affects temperature, fuel moisture)
var elevation = dem.rename('elevation');

var terrain = slope.addBands(aspectSin).addBands(aspectCos)
  .addBands(tpi).addBands(ruggedness).addBands(elevation);

// ═══════════════════════════════════════════════════════════════════
// 2. Fuel / Land Cover (NLCD 30m)
// ═══════════════════════════════════════════════════════════════════

// Use NLCD closest to fire year
var nlcd = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD').select('landcover');

// Fuel load proxy (0-1): higher = more burnable fuel
var fuelLoad = nlcd.remap(
  [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95],
  [0, 0, 0.1, 0.05, 0.02, 0.01, 0.05, 0.7, 0.9, 0.8, 0.6, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3, 0.2, 0.6, 0.4]
).rename('fuel_load');

// Firebreak indicator: water, developed, barren
var firebreak = nlcd.remap(
  [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95],
  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
).rename('is_firebreak');

// Tree canopy cover (%)
var canopy = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD')
  .select('percent_tree_canopy')
  .rename('canopy_cover_pct');

var fuel = fuelLoad.addBands(firebreak).addBands(canopy);

// ═══════════════════════════════════════════════════════════════════
// 3. LANDFIRE Vegetation (30m)
// ═══════════════════════════════════════════════════════════════════

var evt = ee.Image('LANDFIRE/Vegetation/EVT/v1_4_0')
  .select('EVT').rename('vegetation_type');
var evc = ee.Image('LANDFIRE/Vegetation/EVC/v1_4_0')
  .select('EVC').rename('vegetation_cover');
var evh = ee.Image('LANDFIRE/Vegetation/EVH/v1_4_0')
  .select('EVH').rename('vegetation_height');

var landfire = evt.addBands(evc).addBands(evh);

// ═══════════════════════════════════════════════════════════════════
// 4. Fire History (MTBS 30m)
// ═══════════════════════════════════════════════════════════════════

var mtbs = ee.ImageCollection('USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1')
  .filterDate('2000-01-01', FIRE_YEAR + '-01-01')
  .filterBounds(AOI);

// Years since last burn (areas burned <5 years ago have less fuel)
var lastBurnYear = mtbs.map(function(img) {
  var year = ee.Date(img.get('system:time_start')).get('year');
  var burned = img.select('Severity').gte(2).and(img.select('Severity').lte(4));
  return burned.multiply(year).selfMask().rename('burn_year');
}).max();

var yearsSinceBurn = ee.Image(FIRE_YEAR).subtract(lastBurnYear)
  .unmask(99)  // 99 = no burn in record
  .rename('years_since_burn');

// Burn count in record (fire return frequency)
var burnCount = mtbs.map(function(img) {
  return img.select('Severity').gte(2).and(img.select('Severity').lte(4))
    .selfMask().rename('burned');
}).count().unmask(0).rename('burn_count');

var fireHistory = yearsSinceBurn.addBands(burnCount);

// ═══════════════════════════════════════════════════════════════════
// 5. Roads as Firebreaks (TIGER Census)
// ═══════════════════════════════════════════════════════════════════

var roads = ee.FeatureCollection('TIGER/2016/Roads').filterBounds(AOI);

// Rasterize road presence
var roadPresence = roads.reduceToImage({
  properties: ['linearid'],
  reducer: ee.Reducer.count()
}).gt(0).unmask(0).rename('has_road');

// ═══════════════════════════════════════════════════════════════════
// Export combined static features
// ═══════════════════════════════════════════════════════════════════

var allStatic = terrain
  .addBands(fuel)
  .addBands(fireHistory)
  .addBands(roadPresence);

// Resample to GOES grid
Export.image.toDrive({
  image: allStatic,
  description: FIRE_NAME + '_' + FIRE_YEAR + '_StaticFeatures',
  folder: 'fire_pipeline',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// Also export LANDFIRE separately (large categorical rasters)
Export.image.toDrive({
  image: landfire,
  description: FIRE_NAME + '_' + FIRE_YEAR + '_LANDFIRE',
  folder: 'fire_pipeline',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

print('Static feature bands:', allStatic.bandNames());
print('LANDFIRE bands:', landfire.bandNames());
