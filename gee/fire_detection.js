// NOTE: This script is for interactive exploration in the GEE Code Editor.
// For batch processing, use the Python API: `wildfire download <fire_name>`

/**
 * Multi-Source Fire Detection Pipeline
 *
 * Improvements over GOFER:
 * 1. DQF cloud masking — clouds marked as "unknown" not "no fire"
 * 2. Multi-source fusion — GOES + VIIRS + MODIS
 * 3. Full fire characterization — confidence, FRP, sub-pixel area, temperature
 * 4. Uses FDCC (CONUS, 5-min) instead of FDCF (10-min) for 2x temporal density
 * 5. Configurable confidence mapping and thresholds
 * 6. Exports quality flags for downstream filtering
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
var START_UTC = ee.Date(ui.url.get('start', '2019-10-24T00:00:00Z'));
var N_HOURS = parseInt(ui.url.get('hours', '160'), 10);
var EXPORT_SCALE = parseInt(ui.url.get('scale', '2004'), 10);
var EXPORT_CRS = 'EPSG:3857';

// ═══════════════════════════════════════════════════════════════════
// Data Sources
// ═══════════════════════════════════════════════════════════════════

// GOES-17 (West) decommissioned Jan 2023, replaced by GOES-18
// Use GOES-17 for fires before 2023, GOES-18 for 2023+
var GOES_WEST_ID = FIRE_YEAR < 2023 ? '17' : '18';

var goes16_conus = ee.ImageCollection('NOAA/GOES/16/FDCC');
var goes17_conus = ee.ImageCollection('NOAA/GOES/' + GOES_WEST_ID + '/FDCC');
var goes16_full = ee.ImageCollection('NOAA/GOES/16/FDCF');
var goes17_full = ee.ImageCollection('NOAA/GOES/' + GOES_WEST_ID + '/FDCF');

// VIIRS daily fire (science quality, 1km gridded)
var viirs_vnp14 = ee.ImageCollection('NASA/VIIRS/002/VNP14A1');

// MODIS daily fire (Terra + Aqua)
var modis_terra = ee.ImageCollection('MODIS/061/MOD14A1');
var modis_aqua = ee.ImageCollection('MODIS/061/MYD14A1');

// ═══════════════════════════════════════════════════════════════════
// GOES Fire Mask → Confidence Mapping
// ═══════════════════════════════════════════════════════════════════
//
// GOES FDCC/FDCF Mask band encodes fire detection category:
//   10/30 = Processed fire       → 1.0 (highest confidence)
//   11/31 = Saturated fire       → 1.0 (sensor saturated, very hot)
//   12/32 = Cloud contaminated   → 0.8 (fire seen through partial cloud)
//   13/33 = High probability     → 0.5
//   14/34 = Medium probability   → 0.3
//   15/35 = Low probability      → 0.1 (highest false alarm rate)
//
// Codes 30-35 are temporally filtered (2+ detections in 12h window).
// The .mod(10) strips the temporal filter offset, treating both equally.
//
// IMPROVEMENT: We also export whether detection was temporally filtered,
// since filtered detections are more reliable.

var goesFireConfidence = function(image) {
  var mask = image.select('Mask');
  var dqf = image.select('DQF');

  // Fire pixels: mask codes 10-15 (instantaneous) or 30-35 (temporally filtered)
  var isFirePixel = mask.gte(10).and(mask.lte(35));

  // Convert mask to confidence
  var category = mask.mod(10);  // 0=processed, 1=saturated, 2=cloud, 3=high, 4=med, 5=low
  var confidence = category.expression(
    '(c == 0) * 1.0 + (c == 1) * 1.0 + (c == 2) * 0.8 + ' +
    '(c == 3) * 0.5 + (c == 4) * 0.3 + (c == 5) * 0.1',
    {c: category}
  ).updateMask(isFirePixel).rename('fire_confidence');

  // Was this a temporally filtered detection? (more reliable)
  var isTemporallyFiltered = mask.gte(30).and(mask.lte(35))
    .updateMask(isFirePixel).rename('temporal_filtered');

  // DQF quality flags
  // 0 = good fire pixel
  // 1 = good fire-free land
  // 2 = CLOUD (key improvement: mark as unknown, not negative)
  // 3 = invalid (sunglint, bad surface, off-earth, missing)
  // 4 = bad input data
  // 5 = algorithm failure
  var cloudFlag = dqf.eq(2).rename('is_cloud');
  var validFlag = dqf.lte(1).rename('is_valid');  // only 0 and 1 are usable

  // Fire Radiative Power (MW) — only for good-quality fire pixels
  var frp = image.select('Power')
    .updateMask(isFirePixel.and(dqf.eq(0)))
    .rename('frp_mw');

  // Sub-pixel fire area (m²) and temperature (K)
  var area = image.select('Area')
    .updateMask(isFirePixel.and(dqf.eq(0)))
    .rename('fire_area_m2');
  var temp = image.select('Temp')
    .updateMask(isFirePixel.and(dqf.eq(0)))
    .rename('fire_temp_k');

  return confidence
    .addBands(frp)
    .addBands(area)
    .addBands(temp)
    .addBands(cloudFlag)
    .addBands(validFlag)
    .addBands(isTemporallyFiltered)
    .copyProperties(image, ['system:time_start']);
};

// ═══════════════════════════════════════════════════════════════════
// VIIRS Confidence Mapping
// ═══════════════════════════════════════════════════════════════════

var viirsFireConfidence = function(image) {
  var fireMask = image.select('FireMask');
  // FireMask: 7=low, 8=nominal, 9=high confidence fire
  var confidence = fireMask.expression(
    '(fm == 7) * 0.3 + (fm == 8) * 0.7 + (fm == 9) * 1.0',
    {fm: fireMask}
  ).rename('viirs_confidence');

  var frp = image.select('MaxFRP').rename('viirs_frp');
  var hasFire = fireMask.gte(7).rename('viirs_has_fire');

  return confidence.addBands(frp).addBands(hasFire)
    .copyProperties(image, image.propertyNames());
};

// ═══════════════════════════════════════════════════════════════════
// MODIS Confidence Mapping
// ═══════════════════════════════════════════════════════════════════

var modisFireConfidence = function(image) {
  var fireMask = image.select('FireMask');
  var confidence = fireMask.expression(
    '(fm == 7) * 0.3 + (fm == 8) * 0.7 + (fm == 9) * 1.0',
    {fm: fireMask}
  ).rename('modis_confidence');

  // FRP: multiply by 0.1 to get MW
  var frp = image.select('MaxFRP').multiply(0.1).rename('modis_frp');
  var hasFire = fireMask.gte(7).rename('modis_has_fire');

  return confidence.addBands(frp).addBands(hasFire)
    .copyProperties(image, image.propertyNames());
};

// ═══════════════════════════════════════════════════════════════════
// Hourly Multi-Source Fusion
// ═══════════════════════════════════════════════════════════════════

var getHourlyFusedFire = function(hourStart, hourEnd) {
  // ---- GOES (primary: hourly temporal backbone) ----
  var goes_e = goes16_conus.filterDate(hourStart, hourEnd).filterBounds(AOI);
  var goes_w = goes17_conus.filterDate(hourStart, hourEnd).filterBounds(AOI);

  // Fallback to full disk if CONUS empty
  goes_e = ee.ImageCollection(ee.Algorithms.If(
    goes_e.size().gt(0), goes_e,
    goes16_full.filterDate(hourStart, hourEnd).filterBounds(AOI)
  ));
  goes_w = ee.ImageCollection(ee.Algorithms.If(
    goes_w.size().gt(0), goes_w,
    goes17_full.filterDate(hourStart, hourEnd).filterBounds(AOI)
  ));

  // Max confidence across all GOES images in this hour
  var goesE_processed = goes_e.map(goesFireConfidence);
  var goesW_processed = goes_w.map(goesFireConfidence);

  // IMPROVEMENT: Use MAX (not min) for combining East+West
  // GOFER used min which suppressed real detections.
  // Max preserves the best detection from either satellite.
  var goes_all = goesE_processed.merge(goesW_processed);

  var goesConf = goes_all.select('fire_confidence').max()
    .unmask(0).rename('goes_confidence');
  var goesFRP = goes_all.select('frp_mw').max()
    .unmask(0).rename('goes_frp');
  var goesArea = goes_all.select('fire_area_m2').max()
    .unmask(0).rename('goes_fire_area');
  var goesTemp = goes_all.select('fire_temp_k').max()
    .unmask(0).rename('goes_fire_temp');

  // Cloud mask: pixel is cloudy if ANY image in this hour flagged it as cloud
  // and NO image detected fire (fire through cloud is still fire)
  var anyCloud = goes_all.select('is_cloud').max().unmask(0);
  var anyFire = goesConf.gt(0);
  var isCloudNotFire = anyCloud.and(anyFire.not()).rename('is_cloud');

  // Valid observation: at least one good-quality image saw this pixel
  var anyValid = goes_all.select('is_valid').max().unmask(0).rename('is_valid');

  // ---- VIIRS (supplementary: +/- 12h window, 375m-1km) ----
  var viirs_window_start = hourStart.advance(-12, 'hour');
  var viirs_window_end = hourEnd.advance(12, 'hour');

  var viirs = viirs_vnp14
    .filterDate(viirs_window_start, viirs_window_end)
    .filterBounds(AOI)
    .map(viirsFireConfidence);

  var viirsConf = ee.Image(ee.Algorithms.If(
    viirs.size().gt(0),
    viirs.select('viirs_confidence').max().unmask(0),
    ee.Image(0).rename('viirs_confidence')
  ));

  var viirsHasFire = ee.Image(ee.Algorithms.If(
    viirs.size().gt(0),
    viirs.select('viirs_has_fire').max().unmask(0),
    ee.Image(0).rename('viirs_has_fire')
  ));

  // ---- MODIS (supplementary: +/- 12h window, 1km) ----
  var modis = modis_terra
    .filterDate(viirs_window_start, viirs_window_end)
    .filterBounds(AOI)
    .map(modisFireConfidence)
    .merge(
      modis_aqua
        .filterDate(viirs_window_start, viirs_window_end)
        .filterBounds(AOI)
        .map(modisFireConfidence)
    );

  var modisConf = ee.Image(ee.Algorithms.If(
    modis.size().gt(0),
    modis.select('modis_confidence').max().unmask(0),
    ee.Image(0).rename('modis_confidence')
  ));

  // ---- Fused confidence ----
  // When LEO sensors available: weighted combination
  // When only GOES: use GOES alone
  var hasLEO = viirsConf.gt(0).or(modisConf.gt(0));

  var fusedWithLEO = goesConf.multiply(0.3)
    .add(viirsConf.multiply(0.5))
    .add(modisConf.multiply(0.2));

  var fusedConfidence = fusedWithLEO
    .where(hasLEO.not(), goesConf)
    .rename('fused_confidence');

  // ---- Observation quality mask ----
  // 1 = valid observation (can be used for training)
  // 0 = cloud or invalid (should be EXCLUDED from loss, not labeled negative)
  var observationValid = anyValid.and(isCloudNotFire.not()).rename('observation_valid');

  return fusedConfidence
    .addBands(goesConf)
    .addBands(goesFRP)
    .addBands(goesArea)
    .addBands(goesTemp)
    .addBands(isCloudNotFire)
    .addBands(observationValid)
    .addBands(hasLEO.rename('has_leo_confirmation'));
};

// ═══════════════════════════════════════════════════════════════════
// Export: Hourly multi-band stacks
// ═══════════════════════════════════════════════════════════════════

// Build hourly image collection
var hours = ee.List.sequence(0, N_HOURS - 1);

var hourlyCollection = ee.ImageCollection(hours.map(function(h) {
  var hourStart = START_UTC.advance(h, 'hour');
  var hourEnd = START_UTC.advance(ee.Number(h).add(1), 'hour');
  return getHourlyFusedFire(hourStart, hourEnd)
    .set('timeStep', ee.Number(h).add(1))
    .set('hour_start', hourStart);
}));

// Export fused confidence as multi-band image (like GOFER format)
var confBands = hourlyCollection.select('fused_confidence').toBands();
var confBandNames = hours.map(function(h) {
  return ee.String('h_').cat(ee.Number(h).add(10001).toInt());
});

Export.image.toAsset({
  image: confBands.rename(confBandNames),
  description: FIRE_NAME + '_' + FIRE_YEAR + '_FusedConf',
  assetId: 'projects/your-project/fire_pipeline/' + FIRE_NAME + '_' + FIRE_YEAR + '_FusedConf',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// Export observation validity mask (critical: tells model which pixels to include in loss)
var validBands = hourlyCollection.select('observation_valid').toBands();

Export.image.toAsset({
  image: validBands.rename(confBandNames),
  description: FIRE_NAME + '_' + FIRE_YEAR + '_ObsValid',
  assetId: 'projects/your-project/fire_pipeline/' + FIRE_NAME + '_' + FIRE_YEAR + '_ObsValid',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// Export cloud mask (for analysis)
var cloudBands = hourlyCollection.select('is_cloud').toBands();

Export.image.toAsset({
  image: cloudBands.rename(confBandNames),
  description: FIRE_NAME + '_' + FIRE_YEAR + '_CloudMask',
  assetId: 'projects/your-project/fire_pipeline/' + FIRE_NAME + '_' + FIRE_YEAR + '_CloudMask',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

// Export GOES FRP
var frpBands = hourlyCollection.select('goes_frp').toBands();

Export.image.toAsset({
  image: frpBands.rename(confBandNames),
  description: FIRE_NAME + '_' + FIRE_YEAR + '_FRP',
  assetId: 'projects/your-project/fire_pipeline/' + FIRE_NAME + '_' + FIRE_YEAR + '_FRP',
  region: AOI,
  scale: EXPORT_SCALE,
  crs: EXPORT_CRS,
  maxPixels: 1e10
});

print('Hourly collection size:', hourlyCollection.size());
print('First hour bands:', hourlyCollection.first().bandNames());
print('AOI:', AOI);
