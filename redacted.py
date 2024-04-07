#!/usr/bin/env python3
# Interactive Sounding Plotter
# Created 17 May 2023 by Sam Gardner <sam@wx4stg.com>


import airportsdata
import requests
from metpy.units import units
import metpy.calc as mpcalc
from metpy import plots
from metpy import constants
from metpy.io import add_station_lat_lon, parse_metar_to_dataframe
from ecape.calc import calc_ecape
from os import path
from pathlib import Path
import numpy as np
from datetime import datetime as dt
import pandas as pd
import xarray as xr
from cartopy import crs as ccrs
from cartopy import feature as cfeat

import holoviews as hv
hv.extension('bokeh')



def makeSoundingDataset(profileData, icao=None, when=None, selectedParcel="sb"):
    # sort by decreasing pressure
    profileData = profileData.sort_values(by="LEVEL", ascending=False).reset_index(drop=True)
    if "HGHT" in profileData.keys():
        if profileData["HGHT"].is_monotonic_increasing == False:
            profileData = profileData[profileData["HGHT"] >= profileData["HGHT"].cummax()]
    if "WSPD" in profileData.keys() and "WDIR" in profileData.keys():
        profileData["u"], profileData["v"] = mpcalc.wind_components((profileData.WSPD.values * units.kt), (profileData.WDIR.values * units.deg))
    # Create xarray dataset from pandas df
    soundingDS = xr.Dataset.from_dataframe(profileData)
    if "LAT" in profileData.keys():
        # Add balloon path, if available
        soundingDS["LAT"] = soundingDS.LAT * units.degree
        soundingDS["LON"] = soundingDS.LON * units.degree
    elif icao is not None:
        # Get lat/lon from airport code, if provided
        if len(icao) == 3:
            icao = "K"+icao.upper()
        icaoDF = pd.DataFrame({"station" : icao}, index=[0])
        icaoDF = add_station_lat_lon(icaoDF)
        soundingDS.attrs["icao"] = icao
        soundingDS.attrs["LAT"] = icaoDF["latitude"].values[0] * units.degree
        soundingDS.attrs["LON"] = icaoDF["longitude"].values[0] * units.degree
    
    soundingDS["LEVEL_unitless"] = soundingDS.LEVEL
    startLevel = (soundingDS.LEVEL.data[0] // 1)
    endLevel = (soundingDS.LEVEL.data[-1] // 1)
    soundingDS = soundingDS.swap_dims({"index" : "LEVEL_unitless"})
    soundingDS = soundingDS.drop("index")
    soundingDS = soundingDS.sortby("LEVEL")
    soundingDS = soundingDS.drop_duplicates(dim="LEVEL_unitless", keep="first")
    soundingDS = soundingDS.interp(LEVEL_unitless=np.arange(endLevel, startLevel+.05, 1))
    soundingDS = soundingDS.interpolate_na(dim="LEVEL_unitless")
    soundingDS = soundingDS.sortby("LEVEL_unitless", ascending=False)
    soundingDS["index"] = np.arange(len(soundingDS.LEVEL.data))
    soundingDS = soundingDS.dropna(dim="LEVEL_unitless", how="any")
    
    # Add meteorological data
    soundingDS["LEVEL"] = soundingDS.LEVEL * units.hPa
    soundingDS["TEMP"] = soundingDS.TEMP * units.degC
    soundingDS["DWPT"] = soundingDS.DWPT * units.degC
    soundingDS["u"] = soundingDS.u * units.kt
    soundingDS["v"] = soundingDS.v * units.kt
    if "HGHT" in profileData.keys():
        soundingDS["HGHT"] = soundingDS.HGHT * units.meter
        # Calculate AGL heights if MSL heights are available, assume point 0 is the surface
        if "AGL" not in profileData.keys():
            soundingDS["AGL"] = soundingDS.HGHT - soundingDS.HGHT.data[0]
    soundingDS["WSPD"] = mpcalc.wind_speed(soundingDS.u, soundingDS.v)
    soundingDS["WDIR"] = mpcalc.wind_direction(soundingDS.u, soundingDS.v)
    soundingDS["virtT"] = mpcalc.virtual_temperature_from_dewpoint(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS["RH"] = mpcalc.relative_humidity_from_dewpoint(soundingDS.TEMP, soundingDS.DWPT)
    soundingDS["wetbulb"] = mpcalc.wet_bulb_temperature(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    # Calculate effective inflow layer
    inflowBottom = np.nan
    inflowTop = np.nan
    for i in range(len(soundingDS.LEVEL.data)):
        slicedProfile = soundingDS.isel(LEVEL_unitless=slice(i, len(soundingDS.LEVEL.data)))
        capeProfile = mpcalc.parcel_profile(slicedProfile.LEVEL, slicedProfile.TEMP[0], slicedProfile.DWPT[0]).data
        cape, cinh = mpcalc.cape_cin(slicedProfile.LEVEL.data, slicedProfile.TEMP.data, slicedProfile.DWPT.data, parcel_profile=capeProfile)
        # try:
        #     cape, cinh = mpcalc.cape_cin(slicedProfile.LEVEL.data, slicedProfile.TEMP.data, slicedProfile.DWPT.data, parcel_profile=capeProfile)
        # except ValueError as e:
        #     print(slicedProfile.LEVEL.data)
        #     print(slicedProfile.TEMP.data)
        #     print(slicedProfile.DWPT.data)
        #     testDF = pd.DataFrame({"LEVEL" : slicedProfile.LEVEL.data, "TEMP" : slicedProfile.TEMP.data, "DWPT" : slicedProfile.DWPT.data})
        #     testDF.to_csv("test.csv")
        #     mpcalc.lfc(slicedProfile.LEVEL.data, slicedProfile.TEMP.data, slicedProfile.DWPT.data, parcel_temperature_profile=capeProfile, which="bottom")
        if cape.magnitude >= 100 and cinh.magnitude >= -250:
                inflowTop = soundingDS.LEVEL.data[i]
                if np.isnan(inflowBottom):
                    inflowBottom = soundingDS.LEVEL.data[i]
        else:
            if not np.isnan(inflowBottom):
                break
    if inflowBottom == inflowTop:
        inflowBottom = np.nan
        inflowTop = np.nan
    soundingDS.attrs["inflowBottom"] = inflowBottom
    soundingDS.attrs["inflowTop"] = inflowTop

    # Calculate parcel paths, LCLs, LFCs, ELs, CAPE, CINH
    # surface-based
    sbParcelPath = mpcalc.parcel_profile(soundingDS.LEVEL, soundingDS.TEMP[0], soundingDS.DWPT[0]).data
    soundingDS["sbParcelPath"] = sbParcelPath.to(units.degC)
    soundingDS.attrs["sbLCL"] = mpcalc.lcl(soundingDS.LEVEL[0], soundingDS.virtT[0], soundingDS.DWPT[0])[0]
    soundingDS.attrs["sbLFC"] = mpcalc.lfc(soundingDS.LEVEL, soundingDS.virtT, soundingDS.DWPT, parcel_temperature_profile=sbParcelPath)[0]
    soundingDS.attrs["sbEL"] = mpcalc.el(soundingDS.LEVEL, soundingDS.virtT, soundingDS.DWPT, parcel_temperature_profile=sbParcelPath)[0]
    soundingDS.attrs["sbCAPE"], soundingDS.attrs["sbCINH"] = mpcalc.surface_based_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)

    # most unstable
    initPressure, initTemp, initDewpoint, initIdx = mpcalc.most_unstable_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["mu_initPressure"], soundingDS.attrs["mu_initTemp"], soundingDS.attrs["mu_initDewpoint"] = initPressure, initTemp, initDewpoint
    initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
    muParcelPath = np.empty(soundingDS.LEVEL.data.shape)
    muParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], initTemp, initDewpoint).data.to(units.degK).magnitude
    muParcelPath[:initIdx] = np.nan
    muParcelPath = muParcelPath * units.degK
    soundingDS["muParcelPath"] = muParcelPath.to(units.degC)
    soundingDS.attrs["muLCL"] = mpcalc.lcl(initPressure, initTemp, initDewpoint)[0]
    soundingDS.attrs["muLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=muParcelPath[initIdx:])[0]
    soundingDS.attrs["muEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=muParcelPath[initIdx:])[0]
    soundingDS.attrs["muCAPE"], soundingDS.attrs["muCINH"] = mpcalc.most_unstable_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    
    # 100-hPa mixed layer
    mlParcelPath = np.empty(soundingDS.LEVEL.data.shape)
    if np.nanmax(soundingDS.LEVEL.data) - np.nanmin(soundingDS.LEVEL.data) > 100 * units.hPa:
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
        initIdx = len(soundingDS.where(soundingDS.LEVEL >= initPressure, drop=True).LEVEL.data)
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
        mlParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], initVirtT, initDewpoint)
        mlParcelPath[:initIdx] = np.nan
        mlParcelPath = mlParcelPath * units.degK
        soundingDS["mlParcelPath"] = mlParcelPath.to(units.degC)
        soundingDS.attrs["mlLCL"] = mpcalc.lcl(initPressure, initVirtT, initDewpoint)[0]
        soundingDS.attrs["mlLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=mlParcelPath[initIdx:])[0]
        soundingDS.attrs["mlEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=mlParcelPath[initIdx:])[0]
        soundingDS.attrs["mlCAPE"], soundingDS.attrs["mlCINH"] = mpcalc.mixed_layer_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    else:
        mlParcelPath[:] = np.nan
        mlParcelPath = mlParcelPath * units.degK
        soundingDS["mlParcelPath"] = mlParcelPath.to(units.degC)
        soundingDS.attrs["mlLCL"] = np.nan * units.hPa
        soundingDS.attrs["mlLFC"] = np.nan * units.hPa
        soundingDS.attrs["mlEL"] = np.nan * units.hPa
        soundingDS.attrs["mlCAPE"] = np.nan * units.joule/units.kilogram
        soundingDS.attrs["mlCINH"] = np.nan * units.joule/units.kilogram

    # effective inflow layer
    inflowParcelPath = np.empty(soundingDS.LEVEL.data.shape)
    if not np.isnan(inflowBottom):
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_start_pressure=inflowBottom, depth=(inflowBottom - inflowTop))
        initIdx = soundingDS.where(soundingDS.LEVEL >= initPressure, drop=True).index.data[0]
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
        inflowParcelPath[initIdx:] = mpcalc.parcel_profile(soundingDS.LEVEL[initIdx:], soundingDS.TEMP[initIdx], soundingDS.DWPT[initIdx]).data.to(units.degK).magnitude
        inflowParcelPath[:initIdx] = np.nan
        inflowParcelPath = inflowParcelPath * units.degK
        soundingDS.attrs["inLCL"] = mpcalc.lcl(initPressure, initVirtT, initDewpoint)[0]
        soundingDS.attrs["inLFC"] = mpcalc.lfc(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=inflowParcelPath[initIdx:])[0]
        soundingDS.attrs["inEL"] = mpcalc.el(soundingDS.LEVEL[initIdx:], soundingDS.virtT[initIdx:], soundingDS.DWPT[initIdx:], parcel_temperature_profile=inflowParcelPath[initIdx:])[0]
        soundingDS.attrs["inCAPE"], soundingDS.attrs["inCINH"] = mpcalc.mixed_layer_cape_cin(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, parcel_start_pressure=inflowBottom, depth=(inflowBottom - inflowTop))
    else:
        inflowParcelPath[:] = np.nan
        inflowParcelPath = inflowParcelPath * units.degK
        soundingDS.attrs["inLCL"] = np.nan * units.hPa
        soundingDS.attrs["inLFC"] = np.nan * units.hPa
        soundingDS.attrs["inEL"] = np.nan * units.hPa
        soundingDS.attrs["inCAPE"], soundingDS.attrs["inCINH"] = np.nan * units.joule/units.kilogram, np.nan * units.joule/units.kilogram
    soundingDS["inParcelPath"] = inflowParcelPath.to(units.degC)
    # Cloud Layer heights
    soundingDS.attrs["cloudLayerBottom"] = soundingDS.attrs[selectedParcel+"LCL"]
    soundingDS.attrs["cloudLayerTop"] = soundingDS.attrs[selectedParcel+"EL"]

    # pwat
    soundingDS.attrs["pwat"] = mpcalc.precipitable_water(soundingDS.LEVEL, soundingDS.DWPT).to(units.inch)
    # storm motion
    if np.nanmax(soundingDS.AGL.data) > 6000 * units.meter:
        soundingDS.attrs["bunkers_RM"], soundingDS.attrs["bunkers_LM"], soundingDS.attrs["zeroToSixMean"] = mpcalc.bunkers_storm_motion(soundingDS.LEVEL, soundingDS.u, soundingDS.v, soundingDS.HGHT)
        soundingDS.attrs["corfidi_up"], soundingDS.attrs["corfidi_down"] = mpcalc.corfidi_storm_motion(soundingDS.LEVEL, soundingDS.u, soundingDS.v, soundingDS.HGHT)
        # SRH
        soundingDS.attrs["RM_SRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=3000*units.meter, storm_u=soundingDS.bunkers_RM[0], storm_v=soundingDS.bunkers_RM[1])[2]
        soundingDS.attrs["MW_SRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=3000*units.meter, storm_u=soundingDS.zeroToSixMean[0], storm_v=soundingDS.zeroToSixMean[1])[2]
        soundingDS.attrs["LM_SRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=3000*units.meter, storm_u=soundingDS.bunkers_LM[0], storm_v=soundingDS.bunkers_LM[1])[2]
        # RH
        soundingDS.attrs["LL_RH"] = soundingDS.where(soundingDS.LEVEL >= soundingDS.LEVEL[0] - 100*units.hPa, drop=True).RH.data.mean()
        soundingDS.attrs["ML_RH"] = soundingDS.where(soundingDS.LEVEL >= soundingDS.LEVEL[0] - 350*units.hPa, drop=True).RH.data.mean()
        # which bunkers is favored
        if soundingDS.attrs["RM_SRH"] >= soundingDS.attrs["LM_SRH"]:
            soundingDS.attrs["favored_motion"] = "RM"
        else:
            soundingDS.attrs["favored_motion"] = "LM"
    else:
        soundingDS.attrs["bunkers_RM"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["bunkers_LM"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["zeroToSixMean"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["corfidi_up"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["corfidi_down"] = (np.nan * units.knot, np.nan * units.knot)
        soundingDS.attrs["RM_SRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["MW_SRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["LM_SRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["LL_RH"] = np.nan * units.percent
        soundingDS.attrs["ML_RH"] = np.nan * units.percent
        soundingDS.attrs["favored_motion"] = None
    
    if np.nanmax(soundingDS.AGL.data) > 1000 * units.meter:
        soundingDS.attrs["sfc_to_one_shear"] = mpcalc.bulk_shear(soundingDS.LEVEL, soundingDS.u, soundingDS.v, depth=1000 * units.meter)
    else:
        soundingDS.attrs["sfc_to_one_shear"] = [np.nan * units.knot, np.nan * units.knot]
    if np.nanmax(soundingDS.AGL.data) > 6000 * units.meter:
        soundingDS.attrs["sfc_to_six_shear"] = mpcalc.bulk_shear(soundingDS.LEVEL, soundingDS.u, soundingDS.v, depth=6000 * units.meter)
    else:
        soundingDS.attrs["sfc_to_six_shear"] = [np.nan * units.knot, np.nan * units.knot]
    if np.nanmax(soundingDS.AGL.data) > 8000 * units.meter:
        soundingDS.attrs["sfc_to_eight_shear"] = mpcalc.bulk_shear(soundingDS.LEVEL, soundingDS.u, soundingDS.v, depth=8000 * units.meter)
    else:
        soundingDS.attrs["sfc_to_eight_shear"] = [np.nan * units.knot, np.nan * units.knot]

    # AGL versions of the pressure levels
    for key, value in soundingDS.attrs.copy().items():
        if "LCL" in key or "LFC" in key or "EL" in key or "inflow" in key or "cloudLayer" in key:
            if np.isnan(value):
                soundingDS.attrs[key+"_agl"] = np.nan * units.meter
            else:
                soundingDS.attrs[key+"_agl"] = soundingDS.interp(LEVEL_unitless=value.to(units.hPa).magnitude).AGL.data * units.meter

    # Other assorted params needed for SHARPpy's hazard type decision tree
    soundingDS.attrs["sfc_to_one_LR"] = -((soundingDS.TEMP.data[0] - soundingDS.where(soundingDS.AGL <= 1000 * units.meter, drop=True).TEMP.data[-1])/(soundingDS.AGL.data[0] - soundingDS.where(soundingDS.AGL <= 1000 * units.meter, drop=True).AGL.data[-1]).to(units.km))
    soundingDS.attrs["five_to_seven_LR"] = -((soundingDS.where(soundingDS.LEVEL >= 500 * units.hPa, drop=True).TEMP.data[-1] - soundingDS.where(soundingDS.LEVEL >= 700 * units.hPa, drop=True).TEMP.data[-1])/(soundingDS.where(soundingDS.LEVEL >= 500 * units.hPa, drop=True).HGHT.data[-1] - soundingDS.where(soundingDS.LEVEL >= 700 * units.hPa, drop=True).HGHT.data[-1]).to(units.km))
    if np.min(soundingDS.TEMP.data) < 0 * units.degC:
        soundingDS.attrs["freezing_level_agl"] = soundingDS.where(soundingDS.TEMP <= 0 * units.degC, drop=True).AGL.data[0]
    else: 
        soundingDS.attrs["freezing_level_agl"] = np.nan * units.meter
    if soundingDS.favored_motion is not None:
        soundingDS.attrs["favored1kmSRH"] = mpcalc.storm_relative_helicity(soundingDS.AGL, soundingDS.u, soundingDS.v, bottom=0*units.meter, depth=1000*units.meter, storm_u=soundingDS.attrs["bunkers_"+soundingDS.favored_motion][0], storm_v=soundingDS.attrs["bunkers_"+soundingDS.favored_motion][1])[2]
        soundingDS.attrs["fixed_stp"] = mpcalc.significant_tornado(soundingDS.sbCAPE, soundingDS.sbLCL_agl, soundingDS.favored1kmSRH, mpcalc.wind_speed(*soundingDS.sfc_to_six_shear)).to_base_units().magnitude[0]
        cinTerm = 1
        if soundingDS.mlCINH > -50 * units.joule/units.kilogram:
            cinTerm = 1
        elif soundingDS.mlCINH < -200 * units.joule/units.kilogram:
            cinTerm = 0
        else:
            cinTerm = ((soundingDS.mlCINH.magnitude + 200.) / 150.)
        soundingDS.attrs["effective_stp"] = (soundingDS.fixed_stp * cinTerm)
    else:
        soundingDS.attrs["favored1kmSRH"] = np.nan * units.meter**2/units.second**2
        soundingDS.attrs["fixed_stp"] = 0 * units.dimensionless
        soundingDS.attrs["effective_stp"] = 0 * units.dimensionless
    if not np.isnan(soundingDS.attrs["sfc_to_six_shear"][0]):
        shearMag = mpcalc.wind_speed(*soundingDS.attrs["sfc_to_six_shear"])
    layerForCalc = soundingDS.where((soundingDS.LEVEL <= soundingDS.inflowBottom) & (soundingDS.LEVEL >= soundingDS.inflowTop), drop=True)
    if len(layerForCalc.LEVEL.data) > 0 and soundingDS.attrs["favored_motion"] is not None:
        bottom = layerForCalc.AGL.data[0]
        top = layerForCalc.AGL.data[-1]
        depth = (top - bottom)
        favoredEILSRH = mpcalc.storm_relative_helicity(layerForCalc.AGL, layerForCalc.u, layerForCalc.v, bottom=bottom, depth=depth, storm_u=soundingDS.attrs["bunkers_"+soundingDS.attrs["favored_motion"]][0], storm_v=soundingDS.attrs["bunkers_"+soundingDS.attrs["favored_motion"]][1])[2]
        soundingDS.attrs["scp"] = mpcalc.supercell_composite(soundingDS.muCAPE, favoredEILSRH, shearMag)[0]
    else:
        soundingDS.attrs["scp"] = 0 * units.dimensionless
    soundingDS.attrs["k"] = mpcalc.k_index(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["totaltotals"] = mpcalc.total_totals_index(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    soundingDS.attrs["sweat"] = mpcalc.sweat_index(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT, soundingDS.WSPD, soundingDS.WDIR)[0]
    try:
        _, _, soundingDS.attrs["convT"] = mpcalc.ccl(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    except IndexError as e:
        print(e)
        soundingDS.attrs["convT"] = np.nan * units.degC
    # DCAPE
    if np.nanmax(soundingDS.LEVEL.data) > 700 * units.hPa and np.nanmin(soundingDS.LEVEL.data) < 500 * units.hPa:
        soundingDS.attrs["dcape"], soundingDS["dcape_levels"], soundingDS.attrs["dcape_profile"]  = mpcalc.down_cape(soundingDS.LEVEL, soundingDS.TEMP, soundingDS.DWPT)
    else:
        soundingDS.attrs["dcape"] = 0 * units.joule/units.kilogram
        dcapeLevels = np.empty(soundingDS.LEVEL.data.shape)
        dcapeLevels[:] = np.nan
        soundingDS["dcape_levels"] = dcapeLevels
        soundingDS.attrs["dcape_profile"] = [np.nan * units.degC]

    return soundingDS
    

def readACARS(acarsDatasetPath):
    acarsDataset = xr.open_dataset(acarsDatasetPath)
    air = airportsdata.load("IATA")
    for i in range(len(acarsDataset.recNum)):
        thisSounding = acarsDataset.isel(recNum=i)
        thisSoundingICAO = air[bytes(thisSounding.profileAirport.data).decode("utf-8")[:3]]["icao"]
        thisSoundingTime = pd.to_datetime(pd.Timestamp(thisSounding.profileTime.data.item()))
        saveFilePath = path.join(sys.argv[2], thisSoundingICAO, thisSoundingTime.strftime("%Y"), thisSoundingTime.strftime("%m"), thisSoundingTime.strftime("%d"), thisSoundingTime.strftime("%H%M.png"))
        if path.exists(saveFilePath):
            continue
        metarsRaw = requests.get(f"https://www.aviationweather.gov/metar/data?ids={thisSoundingICAO}&format=raw&hours=12&taf=off&layout=on").content.decode("utf-8").split(">\n<code>")
        if "No METAR found for" in metarsRaw:
            continue
        allMetars = None
        for metar in metarsRaw:
            metarStr = metar.replace("</code><br/", "").split(">\n<")[0]
            if metarStr.startswith(thisSoundingICAO):
                metar = parse_metar_to_dataframe(metarStr)
                if allMetars is None:
                    allMetars = metar
                else:
                    allMetars = pd.concat([allMetars, metar])
        if allMetars is None:
            continue
        allMetars["timedelta"] = allMetars["date_time"].apply(lambda x: abs((x - thisSoundingTime).total_seconds()))
        closestMetar = allMetars.sort_values("timedelta").sort_values("timedelta").iloc[0]
        altitudes = [closestMetar["elevation"] * units("m")]
        temperatures = [closestMetar["air_temperature"] * units("degC")]
        dewpoints = [closestMetar["dew_point_temperature"] * units("degC")]
        windDirection = [closestMetar["wind_direction"] * units.deg]
        windSpeed = [closestMetar["wind_speed"] * units("kt")]
        airportBarometer = mpcalc.altimeter_to_station_pressure(closestMetar["altimeter"] * units("inHg"), closestMetar["elevation"] * units("m")).to("hPa")
        pressure = [airportBarometer]
        virtTemp = [mpcalc.virtual_temperature(closestMetar["air_temperature"] * units("degC"), mpcalc.mixing_ratio_from_specific_humidity(mpcalc.specific_humidity_from_dewpoint(airportBarometer, closestMetar["dew_point_temperature"] * units("degC"))))]
        for ii in range(0, len(thisSounding.altitude)):
            if np.isnan(thisSounding.altitude.data[ii]) or np.isnan(thisSounding.temperature.data[ii]):# or np.isnan(thisSounding.dewpoint.data[ii]) or np.isnan(thisSounding.windDir.data[ii]) or np.isnan(thisSounding.windSpeed.data[ii]):
                continue
            if thisSounding.altitude.data[ii] * units.meter in altitudes:
                continue
            thisTemp = thisSounding.temperature.data[ii] * units("K")
            lastTemp = temperatures[-1]
            thisAlt = thisSounding.altitude.data[ii] * units("m")
            lastAlt = altitudes[-1]
            lastPressure = pressure[-1]
            
            thisPressureFactor = (np.abs(lastTemp*lastAlt-thisTemp*thisAlt)/np.abs(lastTemp*(2*lastAlt-thisAlt)-thisTemp*lastAlt))**((9.81*(lastAlt-thisAlt))/(287*(lastTemp-thisTemp))).magnitude
            topPressure = lastPressure*thisPressureFactor**(-1)
            
            if topPressure not in pressure:
                temperatures.append(thisTemp)
                dewpoints.append(thisSounding.dewpoint.data[ii] * units("K"))
                windDirection.append(thisSounding.windDir.data[ii] * units.deg)
                windSpeed.append((thisSounding.windSpeed.data[ii] * units("meter / second")).to("kt"))
                altitudes.append(thisAlt)
                pressure.append(topPressure)
                virtTemp.append(mpcalc.virtual_temperature(thisTemp, mpcalc.mixing_ratio_from_specific_humidity(mpcalc.specific_humidity_from_dewpoint(lastPressure, dewpoints[-1]))))
        
        pressure = np.array([pres.to("hPa").magnitude for pres in pressure])
        temperatures = np.array([temp.to("degC").magnitude for temp in temperatures])
        dewpoints = np.array([dew.to("degC").magnitude for dew in dewpoints])
        windDirection = np.array([windDir.to("deg").magnitude for windDir in windDirection])
        windSpeed = np.array([windSpd.to("kt").magnitude for windSpd in windSpeed])
        altitudes = np.array([alt.to("m").magnitude for alt in altitudes])

        soundingDataFrame = pd.DataFrame({"LEVEL": pressure, "HGHT": altitudes, "TEMP": temperatures, "DWPT": dewpoints, "WDIR": windDirection, "WSPD": windSpeed}).sort_values("LEVEL", ascending=False)
        data = makeSoundingDataset(soundingDataFrame, thisSoundingICAO, thisSoundingTime)
        saveFilePath = path.join(sys.argv[2], thisSoundingICAO, thisSoundingTime.strftime("%Y"), thisSoundingTime.strftime("%m"), thisSoundingTime.strftime("%d"), thisSoundingTime.strftime("%H%M.png"))
        print(thisSoundingICAO, thisSoundingTime)
        plotSounding(data, saveFilePath, thisSoundingICAO, thisSoundingTime, soundingType="ACARS")


def readSharppy(fileName):
    from io import StringIO
    with open(fileName, "r") as f:
        text = f.readlines()
    preamble = text[0]
    whenAndWhere = text[1]
    if whenAndWhere.startswith(" "):
        where = whenAndWhere.split(" ")[1]
    else:
        where = whenAndWhere.split(" ")[0]
    when = dt.strptime(str(dt.utcnow().year)[:2]+whenAndWhere.split(" ")[-1], "%Y%m%d/%H%M\n")
    who = text[2]
    rest = "".join(text[3:])
    data = "".join(rest.split("%RAW%")[1]).split("%END%")[0].replace(" ", "")
    data = pd.read_csv(StringIO(data), sep=",", header=None, names=["LEVEL", "HGHT", "TEMP", "DWPT", "WDIR", "WSPD"])
    data = data.replace(-9999, np.nan)
    for i in range(len(data)):
        if np.isnan(data.iloc[0]["TEMP"]) or np.isnan(data.iloc[0]["DWPT"]) or np.isnan(data.iloc[0]["HGHT"]) or np.isnan(data.iloc[0]["LEVEL"]):
            print(f"Warning: packet {i} is invalid, removing")
            data = data.iloc[1:]
        else:
            break
    data = data.loc[data["LEVEL"] >= 10]
    data = makeSoundingDataset(data, where, when)
    return data, where, when

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: soundingPlot.py <input> <output>")
        exit()
    if not path.exists(sys.argv[1]):
        print("Input file does not exist!")
        exit()
    if sys.argv[1].endswith("acars.nc"):
        readACARS(sys.argv[1])
    else:
        profileData, icao, datetime  = readSharppy(sys.argv[1])
        plotSounding(profileData, sys.argv[2], icao, datetime)

