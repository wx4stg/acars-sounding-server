#!/usr/bin/env python3
# Sam's sounding plotter
# Created 17 May 2023 by Sam Gardner <sam@wx4stg.com>


from metpy.units import units
import metpy.calc as mpcalc
from metpy import plots
from metpy import constants
from metpy.cbook import get_test_data
from ecape.calc import calc_ecape
from matplotlib import pyplot as plt
from matplotlib.table import table
from matplotlib.patches import Polygon
from matplotlib.patheffects import withStroke
from os import path
from pathlib import Path
import numpy as np
from datetime import datetime as dt
import pandas as pd
from cartopy import crs as ccrs
from cartopy import feature as cfeat


def plotParams(profileData, ax):
    pwat = mpcalc.precipitable_water(profileData["LEVEL"].values * units.hPa, profileData["DWPT"].values * units.degC).to("inches")
    pressure = profileData["LEVEL"].values * units.hPa
    temperature = profileData["TEMP"].values * units.degC
    dewpoint = profileData["DWPT"].values * units.degC
    mucape = mpcalc.most_unstable_cape_cin(pressure, temperature, dewpoint)[0]
    if "inEIL" in profileData.keys():
        layerForCalc = profileData.loc[profileData["inEIL"] == 1]
        shearU, shearV = mpcalc.bulk_shear((profileData["LEVEL"].values * units.hPa), (profileData["ukt"].values * units.kt), (profileData["vkt"].values * units.kt), height=(profileData["AGL"].values*units.meter), bottom=0*units.meter, depth=6000 * units.meter)
        shearMag, shearDir = mpcalc.wind_speed(shearU, shearV), mpcalc.wind_direction(shearU, shearV)
    elif "AGL" in profileData.keys():
        layerForCalc = profileData.loc[profileData["AGL"] <= 3000]
        shearU, shearV = mpcalc.bulk_shear((profileData["LEVEL"].values * units.hPa), (profileData["ukt"].values * units.kt), (profileData["vkt"].values * units.kt), height=(profileData["AGL"].values*units.meter), bottom=0*units.meter, depth=6000 * units.meter)
        shearMag, shearDir = mpcalc.wind_speed(shearU, shearV), mpcalc.wind_direction(shearU, shearV)
    elif "HGHT" in profileData.keys():
        profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
        layerForCalc = profileData.loc[profileData["AGL"] <= 3000]
        shearU, shearV = mpcalc.bulk_shear((profileData["LEVEL"].values * units.hPa), (profileData["ukt"].values * units.kt), (profileData["vkt"].values * units.kt), height=(profileData["AGL"].values*units.meter), bottom=0*units.meter, depth=6000 * units.meter)
        shearMag, shearDir = mpcalc.wind_speed(shearU, shearV), mpcalc.wind_direction(shearU, shearV)
    else:
        layerForCalc = profileData.loc[profileData["LEVEL"] >= 700]
        shearU, shearV = mpcalc.bulk_shear((profileData["LEVEL"].values * units.hPa), (profileData["ukt"].values * units.kt), (profileData["vkt"].values * units.kt), height=(profileData["AGL"].values*units.meter), depth=500 * units.hPa)
        shearMag, shearDir = mpcalc.wind_speed(shearU, shearV), mpcalc.wind_direction(shearU, shearV)
    bottom, top = layerForCalc["AGL"].iloc[0], layerForCalc["AGL"].iloc[-1]
    depth = (top - bottom) * units.meter
    rm, lm, mw = mpcalc.bunkers_storm_motion(profileData["LEVEL"].values * units.hPa, profileData["ukt"].values * units.kt, profileData["vkt"].values * units.kt, profileData["HGHT"].values * units.meter)
    RMstormRelativeHelicity = mpcalc.storm_relative_helicity((layerForCalc["AGL"].values*units.meter), (layerForCalc["ums"].values * units.meter/units.sec), (layerForCalc["vms"].values * units.meter/units.sec), bottom=bottom*units.meter, depth=depth, storm_u=rm[0], storm_v=rm[1])[2]
    LMstormRelativeHelicity = mpcalc.storm_relative_helicity((layerForCalc["AGL"].values*units.meter), (layerForCalc["ums"].values * units.meter/units.sec), (layerForCalc["vms"].values * units.meter/units.sec), bottom=bottom*units.meter, depth=depth, storm_u=lm[0], storm_v=lm[1])[2]
    if RMstormRelativeHelicity > LMstormRelativeHelicity:
        scp = mpcalc.supercell_composite(mucape, RMstormRelativeHelicity, shearMag)
    else:
        scp = mpcalc.supercell_composite(mucape, LMstormRelativeHelicity, shearMag)
    
    ax.text(0.5, 0.79, f"PWAT: {pwat.magnitude:.2f}in", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.5, 0.69, f"SCP: {scp[0].magnitude:.1f}", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
    
def plotThermoynamics(profileData, ax, parcelType="sb"):
    if "HGHT" in profileData.keys():
        if "AGL" not in profileData.keys():
            profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
    temperature = profileData["TEMP"].values * units.degC
    dewpoint = profileData["DWPT"].values * units.degC
    pressure = profileData["LEVEL"].values * units.hPa
    msl = profileData["HGHT"].values * units.meter
    temperature3 = profileData["TEMP"].loc[profileData["AGL"] <= 3000].values * units.degC
    dewpoint3 = profileData["DWPT"].loc[profileData["AGL"] <= 3000].values * units.degC
    pressure3 = profileData["LEVEL"].loc[profileData["AGL"] <= 3000].values * units.hPa
    u = profileData["ukt"].values * units.kt
    v = profileData["vkt"].values * units.kt
    virtualTemperature = mpcalc.virtual_temperature_from_dewpoint(pressure, temperature, dewpoint)
    thermodynamicsTableContent = [["Parcel", "CAPE", "CINH", "ECAPE", "LCL", "LFC", "EL", "0->3km CAPE"]]

    variables = [
        ("Surface Based", mpcalc.surface_based_cape_cin),
        ("Mixed Layer", mpcalc.mixed_layer_cape_cin),
        ("Most Unstable", mpcalc.most_unstable_cape_cin)
    ]

    for var in variables:
        caption, cape_func = var
        cape, cin = cape_func(pressure, temperature, dewpoint)
        ecape = calc_ecape(msl, pressure, temperature, mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint), u, v, cape_type=caption.lower().replace(" ", "_"))
        parcel = mpcalc.parcel_profile(pressure, virtualTemperature[0], dewpoint[0])
        lclP = mpcalc.lcl(pressure[0], virtualTemperature[0], dewpoint[0])[0]
        lclm = profileData.loc[(profileData["LEVEL"] - lclP.magnitude).abs().idxmin()]["AGL"] * units.meter
        lfcP = mpcalc.lfc(pressure, virtualTemperature, dewpoint, parcel_temperature_profile=parcel)[0]
        lfcm = profileData.loc[(profileData["LEVEL"] - lfcP.magnitude).abs().idxmin()]["AGL"] * units.meter
        elP = mpcalc.el(pressure, virtualTemperature, dewpoint, parcel_temperature_profile=parcel)[0]
        elm = profileData.loc[(profileData["LEVEL"] - elP.magnitude).abs().idxmin()]["AGL"] * units.meter
        cape3km = cape_func(pressure3, temperature3, dewpoint3)[0]
        thermodynamicsTableContent.append([caption, f"{int(cape.magnitude)} J/kg", f"{int(cin.magnitude)} J/kg", f"{int(ecape.magnitude)} J/kg", f"{int(lclm.magnitude)} m\n{int(lclP.magnitude)} hPa", f"{int(lfcm.magnitude)} m\n{int(lfcP.magnitude)} hPa", f"{int(elm.magnitude)} m\n{int(elP.magnitude)} hPa", f"{int(cape3km.magnitude)} J/kg"])


    if "inEIL" in profileData.keys():
        inflowBottom = profileData.loc[profileData["inEIL"] == 1]["LEVEL"].iloc[0] * units.hPa
        inflowTop = profileData.loc[profileData["inEIL"] == 1]["LEVEL"].iloc[-1] * units.hPa
        caption = "Effective Inflow"
        cape_func = mpcalc.mixed_layer_cape_cin
        cape, cin = cape_func(pressure, temperature, dewpoint, bottom=inflowBottom, depth=(inflowBottom - inflowTop))
        ecape = calc_ecape(msl, pressure, temperature, mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint), u, v, undiluted_cape=cape)
        parcel = mpcalc.mixed_parcel(pressure, temperature, dewpoint, bottom=inflowBottom, depth=(inflowBottom - inflowTop))
        virtT = mpcalc.virtual_temperature_from_dewpoint(parcel[0], parcel[1], parcel[2])
        parcelProfile = mpcalc.parcel_profile(pressure, virtT, parcel[2])
        lclP = mpcalc.lcl(parcel[0], virtT, parcel[2])[0]
        lclm = profileData.loc[(profileData["LEVEL"] - lclP.magnitude).abs().idxmin()]["AGL"] * units.meter
        lfcP = mpcalc.lfc(pressure, virtualTemperature, dewpoint, parcel_temperature_profile=parcelProfile)[0]
        lfcm = profileData.loc[(profileData["LEVEL"] - lfcP.magnitude).abs().idxmin()]["AGL"] * units.meter
        elP = mpcalc.el(pressure, virtualTemperature, dewpoint, parcel_temperature_profile=parcelProfile)[0]
        elm = profileData.loc[(profileData["LEVEL"] - elP.magnitude).abs().idxmin()]["AGL"] * units.meter
        cape3km = cape_func(pressure3, temperature3, dewpoint3, bottom=inflowBottom, depth=(inflowBottom - inflowTop))[0]
        thermodynamicsTableContent.append([caption, f"{int(cape.magnitude)} J/kg", f"{int(cin.magnitude)} J/kg", f"{int(ecape.magnitude)} J/kg", f"{int(lclm.magnitude)} m\n{int(lclP.magnitude)} hPa", f"{int(lfcm.magnitude)} m\n{int(lfcP.magnitude)} hPa", f"{int(elm.magnitude)} m\n{int(elP.magnitude)} hPa", f"{int(cape3km.magnitude)} J/kg"])


    
    thermodynamicsTable = table(ax, bbox=[0, 0, 1, 0.89], cellText=thermodynamicsTableContent, cellLoc="center")


def plotDynamics(profileData, ax):
    dynamicsTableContent = [["Bunkers Right//0-6km Mean Wind//Bunkers Left", "Bulk\nWind\nDifference", "Mean Wind", "Storm\nRelative\nHelicity", "Storm\nRelative\nWind", "Horizontal\nVorticity\n(Streamwise%)"]]
    if "HGHT" in profileData.keys():
        if "AGL" not in profileData.keys():
            profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
        profileDataWithWind = profileData.loc[~np.isnan(profileData["ukt"])].reset_index(drop=True)
        rm, lm, mw = mpcalc.bunkers_storm_motion(profileDataWithWind["LEVEL"].values * units.hPa, profileDataWithWind["ukt"].values * units.kt, profileDataWithWind["vkt"].values * units.kt, profileDataWithWind["HGHT"].values * units.meter)
        if "inEIL" in profileDataWithWind.keys():
            eil = profileDataWithWind.loc[profileDataWithWind["inEIL"] == 1]
            eilBottom, eilTop = eil["AGL"].iloc[0], eil["AGL"].iloc[-1]
        if "inCloud" in profileDataWithWind.keys():
            cloudLayer = profileDataWithWind.loc[profileDataWithWind["inCloud"] == 1]
            cloudLayerBottom, cloudLayerTop = cloudLayer["AGL"].iloc[0], cloudLayer["AGL"].iloc[-1]
        levelsToSample = [(eilBottom, eilTop), (0, 500), (0, 1000), (0, 3000), (0, 6000), (0, 9000), (cloudLayerBottom, cloudLayerTop)]
        for i in range(len(levelsToSample)):
            thisLayerRow = []
            bottom, top = levelsToSample[i]
            depth = (top - bottom)*units.meter
            layerForCalc = profileDataWithWind.loc[(profileDataWithWind["AGL"] >= bottom) & (profileDataWithWind["AGL"] <= top)]
            bottomp, topp = round((((layerForCalc["LEVEL"].values[0]+.0005)//0.001)*.001), 2)*units.hPa, round(((layerForCalc["LEVEL"].values[-1]//0.001)*.001), 2)*units.hPa
            depthp = (bottomp.magnitude - topp.magnitude)//.001*.001*units.hPa
            if i == 0:
                thisLayerRow.append("Effective\nInflow\nLayer")
            elif i == len(levelsToSample)-1:
                thisLayerRow.append("Cloud Layer\n(LCL->EL)")
            elif i == 1:
                thisLayerRow.append(f"SFC->500m")
            else:
                thisLayerRow.append(f"SFC->{int(top/1000)}km")
            shearU, shearV = mpcalc.bulk_shear((layerForCalc["LEVEL"].values * units.hPa), (layerForCalc["ukt"].values * units.kt), (layerForCalc["vkt"].values * units.kt), height=(layerForCalc["AGL"].values*units.meter), bottom=bottomp, depth=depthp)
            shearMag, shearDir = mpcalc.wind_speed(shearU, shearV), mpcalc.wind_direction(shearU, shearV)
            thisLayerRow.append(f"{shearMag.magnitude:.1f}kt//{shearDir.magnitude:.1f}°")
            meanWindu, meanWindv = mpcalc.mean_pressure_weighted((layerForCalc["LEVEL"].values * units.hPa), (layerForCalc["ukt"].values * units.kt), (layerForCalc["vkt"].values * units.kt), height=(layerForCalc["AGL"].values*units.meter), bottom=(bottomp), depth=depthp)
            meanWindMag, meanWindDir = mpcalc.wind_speed(meanWindu, meanWindv), mpcalc.wind_direction(meanWindu, meanWindv)
            thisLayerRow.append(f"{meanWindMag.magnitude:.1f}kt//{meanWindDir.magnitude:.1f}°")
            RMstormRelativeHelicity = mpcalc.storm_relative_helicity((layerForCalc["AGL"].values*units.meter), (layerForCalc["ums"].values * units.meter/units.sec), (layerForCalc["vms"].values * units.meter/units.sec), bottom=bottom*units.meter, depth=depth, storm_u=rm[0], storm_v=rm[1])[2]
            MWstormRelativeHelicity = mpcalc.storm_relative_helicity((layerForCalc["AGL"].values*units.meter), (layerForCalc["ums"].values * units.meter/units.sec), (layerForCalc["vms"].values * units.meter/units.sec), bottom=bottom*units.meter, depth=depth, storm_u=mw[0], storm_v=mw[1])[2]
            LMstormRelativeHelicity = mpcalc.storm_relative_helicity((layerForCalc["AGL"].values*units.meter), (layerForCalc["ums"].values * units.meter/units.sec), (layerForCalc["vms"].values * units.meter/units.sec), bottom=bottom*units.meter, depth=depth, storm_u=lm[0], storm_v=lm[1])[2]
            if top == 3000:
                if RMstormRelativeHelicity >= LMstormRelativeHelicity:
                    rmcolor = "red"
                    lmcolor = "gray"
                else:
                    rmcolor = "gray"
                    lmcolor = "blue"
            thisLayerRow.append(str(int(RMstormRelativeHelicity.magnitude))+"$m^2 s^{-2}$//"+str(int(MWstormRelativeHelicity.magnitude))+"$m^2 s^{-2}$//"+str(int(LMstormRelativeHelicity.magnitude))+"$m^2 s^{-2}$")
            RMsrwU, RMsrwV = (meanWindu-rm[0]), (meanWindv-rm[1])
            RMsrwMag, RMsrwDir = mpcalc.wind_speed(RMsrwU, RMsrwV), mpcalc.wind_direction(RMsrwU, RMsrwV)
            MWsrwU, MWsrwV = (meanWindu-mw[0]), (meanWindu-mw[1])
            MWsrwMag, MWsrwDir = mpcalc.wind_speed(MWsrwU, MWsrwV), mpcalc.wind_direction(MWsrwU, MWsrwV)
            LMsrwU, LMsrwV = (meanWindu-lm[0]), (meanWindv-lm[1])
            LMsrwMag, LMsrwDir = mpcalc.wind_speed(LMsrwU, LMsrwV), mpcalc.wind_direction(LMsrwU, LMsrwV)
            thisLayerRow.append(f"{int(RMsrwMag.magnitude)}kt/{int(RMsrwDir.magnitude)}°//{int(MWsrwMag.magnitude)}kt/{int(MWsrwDir.magnitude)}°//{int(LMsrwMag.magnitude)}kt/{int(LMsrwDir.magnitude)}°")
            
            dvdz = mpcalc.first_derivative((layerForCalc["vms"].values * units.meter/units.sec), x=(layerForCalc["AGL"].values*units.meter))
            dvdz = mpcalc.mean_pressure_weighted((layerForCalc["LEVEL"].values * units.hPa), dvdz, height=(layerForCalc["AGL"].values*units.meter), bottom=bottomp, depth=depthp)[0]
            dudz = mpcalc.first_derivative((layerForCalc["ums"].values * units.meter/units.sec), x=(layerForCalc["AGL"].values*units.meter))
            dudz = mpcalc.mean_pressure_weighted((layerForCalc["LEVEL"].values * units.hPa), dudz, height=(layerForCalc["AGL"].values*units.meter), bottom=bottomp, depth=depthp)[0]
            
            horizVort = ((-dvdz)**2 + (dudz)**2)**0.5
            
            rmstreamwiseVort = ((-dvdz * RMsrwU) + (dudz * RMsrwV))/((RMsrwU**2 + RMsrwV**2)**(0.5))
            rmcrosswiseVort = (horizVort**2 - rmstreamwiseVort**2)**0.5
            rmstreamwisePercent = (rmstreamwiseVort/(rmstreamwiseVort+rmcrosswiseVort))*100
            
            mwstreamwiseVort = ((-dvdz * MWsrwU) + (dudz * MWsrwV))/((MWsrwU**2 + MWsrwV**2)**(0.5))
            mwcrosswiseVort = (horizVort**2 - mwstreamwiseVort**2)**0.5
            mwstreamwisePercent = (mwstreamwiseVort/(mwstreamwiseVort+mwcrosswiseVort))*100

            lmstreamwiseVort = ((-dvdz * LMsrwU) + (dudz * LMsrwV))/((LMsrwU**2 + LMsrwV**2)**(0.5))
            lmcrosswiseVort = (horizVort**2 - lmstreamwiseVort**2)**0.5
            lmstreamwisePercent = np.abs(lmstreamwiseVort/(lmstreamwiseVort+lmcrosswiseVort))*100

            thisLayerRow.append(str(round(horizVort.magnitude, 3))+"$s^{-1}$//"+str(int(rmstreamwisePercent.magnitude))+"%//"+str(int(mwstreamwisePercent.magnitude))+"%//"+str(int(lmstreamwisePercent.magnitude))+"%")
            dynamicsTableContent.append(thisLayerRow)
        dynamicsTable = table(ax, bbox=[0, 0, 1, 1], cellText=np.empty((8,6), dtype=str), cellLoc="center")
        for rowNum in range(len(dynamicsTableContent)):
            for colNum in range(len(dynamicsTableContent[rowNum])):
                if rowNum == 0:
                    if colNum == 0:
                        textArr = dynamicsTableContent[rowNum][colNum].split("//")
                        rm = textArr[0]
                        mw = textArr[1]
                        lm = textArr[2]
                        ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.3+rowNum)/len(dynamicsTableContent)), "Color Key", color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                        ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.4+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="center", va="center", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                        ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.6+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="center", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                        ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.7+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="center", va="top", transform=ax.transAxes, clip_on=False, fontsize=8, zorder=5)
                    else:
                        ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
                elif colNum == 0:
                    if rowNum == 1:
                        color = "teal"
                    elif rowNum == 2:
                        color = "fuchsia"
                    elif rowNum == 3:
                        color = "firebrick"
                    elif rowNum == 4:
                        color = "limegreen"
                    elif rowNum == 5:
                        color = "gold"
                    elif rowNum == 6:
                        color = "darkturquoise"
                    elif rowNum == 7:
                        color = "mediumpurple"
                    else:
                        color = "black"
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], color=color, ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
                elif colNum == 1:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    mag = textArr[0]
                    dir = textArr[1]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mag, color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dir, color="gray", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                elif colNum == 2:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    mag = textArr[0]
                    dir = textArr[1]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mag, color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dir, color="black", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                elif colNum == 3:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    rm = textArr[0]
                    mw = textArr[1]
                    lm = textArr[2]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.4+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.6+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                elif colNum == 4:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    rm = textArr[0]
                    mw = textArr[1]
                    lm = textArr[2]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.4+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.6+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                elif colNum == 5:
                    textArr = dynamicsTableContent[rowNum][colNum].split("//")
                    hv = textArr[0]
                    rm = textArr[1]
                    mw = textArr[2]
                    lm = textArr[3]
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), hv, color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.3+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), rm, color=rmcolor, ha="right", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), mw, color="sienna", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                    ax.text((0.7+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), lm, color=lmcolor, ha="left", va="top", transform=ax.transAxes, clip_on=False, zorder=5)
                else:
                    ax.text((0.5+colNum)/len(dynamicsTableContent[rowNum]), 1-((0.5+rowNum)/len(dynamicsTableContent)), dynamicsTableContent[rowNum][colNum], ha="center", va="center", transform=ax.transAxes, clip_on=False, zorder=5)

def plotPartialThickness(profileData, ax):
    if "HGHT" in profileData.keys():
        if "AGL" not in profileData.keys():
            profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
        midLevel = profileData.loc[profileData["LEVEL"] <= 850].loc[profileData["LEVEL"] >= 700]

        lowLevel = profileData.loc[profileData["LEVEL"] <= 1000].loc[profileData["LEVEL"] >= 850]
        if len(midLevel) >= 1 and len(lowLevel) >= 1:
            midLevelThickness = np.nanmax(midLevel["AGL"].values) - np.nanmin(midLevel["AGL"].values)
            lowLevelThickness = np.nanmax(lowLevel["AGL"].values) - np.nanmin(lowLevel["AGL"].values)
            if midLevelThickness >=1560:
                midLevelThickness = 1560
            if midLevelThickness <= 1525:
                midLevelThickness = 1525
            if lowLevelThickness >= 1315:
                lowLevelThickness = 1315
            if lowLevelThickness <= 1281:
                lowLevelThickness = 1281
        else:
            midLevelThickness = np.nan
            lowLevelThickness = np.nan
        ax.scatter(midLevelThickness, lowLevelThickness, color="gold", edgecolor="black", marker="*", s=125, zorder=10)
    ax.text(0.5, 0, "850-700 hPa Thickness", color="black", ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=6)
    ax.text(0.01, 0.5, "1000-850 hPa Thickness", color="black", ha="left", va="center", transform=ax.transAxes, clip_on=False, zorder=5, rotation=90, fontsize=6)
    snowPoly = Polygon(np.array([
        [0,0],
        [0, 1537.375],
        [1500, 1303],
        [1532, 1298],
        [1537, 1290],
        [1542, 1278],
        [1545, 1260],
        [1755, 0]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(snowPoly)
    ax.text(1527, 1287, "Snow", color="black", ha="left", va="bottom", fontsize=8, zorder=2)
    unknownPoly = Polygon(np.array([
        [0, 20000],
        [0, 1537.375],
        [1500, 1303],
        [1532, 1298],
        [1530, 1312],
        [1538, 1312],
        [1538, 20000]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(unknownPoly)
    ax.text(1525, 1314, "Unknown", color="black", ha="left", va="top", fontsize=8, zorder=2)
    snowRainPoly = Polygon(np.array([
        [1530, 1312],
        [1538, 1312],
        [1543, 1290],
        [1537, 1290],
        [1532, 1298]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(snowRainPoly)
    ax.text(1536, 1300, "Snow +\nRain", color="black", ha="center", va="center", fontsize=8, zorder=2)
    wintryMixPoly = Polygon(np.array([
        [1538, 1312],
        [1550, 1310],
        [1550, 1290],
        [1543, 1290]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(wintryMixPoly)
    ax.text(1545, 1300, "Wintry\nMix", color="black", ha="center", va="center", fontsize=8, zorder=2)
    rainPoly = Polygon(np.array([
        [1538, 1312],
        [1550, 1310],
        [1580, 1314],
        [20000, 3770],
        [20000, 20000],
        [1538, 20000]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(rainPoly)
    ax.text(1554, 1314, "Rain", color="black", ha="right", va="top", fontsize=8, zorder=2)
    snowSleetPoly = Polygon(np.array([
        [1537, 1290],
        [1550, 1290],
        [1550, 1282],
        [1562, 1260],
        [(24742/11), 0],
        [1755, 0],
        [1545, 1260],
        [1542, 1278]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(snowSleetPoly)
    ax.text(1544, 1285, "Snow+\nSleet", color="black", ha="center", va="center", fontsize=8, zorder=2)
    sleetFZRAPoly = Polygon(np.array([
        [1550, 1293],
        [1580, 1290],
        [14480, 0],
        [(24742/11), 0],
        [1562, 1260],
        [1550, 1282],
        [1550, 1290]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(sleetFZRAPoly)
    ax.text(1555, 1285, "Sleet+\nFrz. Rain", color="black", ha="center", va="center", fontsize=8, zorder=2)
    FZRASleetPoly = Polygon(np.array([
        [1550, 1293],
        [1580, 1290],
        [1850, 1263],
        [1580, 1299],
        [1550, 1303]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(FZRASleetPoly)
    ax.text(1555, 1295, "Frz. Rain+\nSleet", color="black", ha="center", va="center", fontsize=8, zorder=2)
    FRZARainPoly = Polygon(np.array([
        [1550, 1310],
        [1580, 1314],
        [20000, 3770],
        [20000, 0],
        [14480, 0],
        [1850, 1263],
        [1580, 1299],
        [1550, 1303]
    ]), closed=True, facecolor="white", edgecolor="black", linewidth=1, zorder=1)
    ax.add_patch(FRZARainPoly)
    ax.text(1555, 1305, "Frz. Rain", color="black", ha="center", va="center", fontsize=8, zorder=2)
    ax.set_xlim([1525, 1560])
    ax.set_ylim([1281, 1315])
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
    ax.grid(color="gray", linestyle="--", linewidth=0.5, zorder=0)
    isSaturatedAnywhere = False
    if "RH" in profileData.keys():
        for top in range(100, 1101, 1):
            bottom = top + 50
            selectedData = profileData.loc[(profileData["LEVEL"] <= bottom) & (profileData["LEVEL"] > top)].reset_index(drop=True)
            if len(selectedData) <= 1:
                continue
            RHinLayer = mpcalc.mean_pressure_weighted(selectedData["LEVEL"].values * units.hPa, selectedData["RH"].values, height=(selectedData["AGL"].values*units.meter), bottom=(selectedData["LEVEL"].values[0]*units.hPa), depth=(selectedData["LEVEL"].values[0]*units.hPa - selectedData["LEVEL"].values[-1]*units.hPa))[0].magnitude
            if RHinLayer >= .8:
                isSaturatedAnywhere = True
                break
    if isSaturatedAnywhere:
        if np.isnan(midLevelThickness) or np.isnan(lowLevelThickness):
            if profileData["TEMP"].iloc[0] >= 5:
                return "sfcRAIN"
            else:
                return "UNKNOWNUNKNOWN"
        elif snowPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SNOW"
        elif unknownPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "UNKNOWN"
        elif snowRainPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SNOW+RAIN"
        elif wintryMixPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "WINTRY MIX"
        elif rainPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "RAIN"
        elif snowSleetPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SNOW+SLEET"
        elif sleetFZRAPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "SLEET+FREEZING RAIN"
        elif FZRASleetPoly.contains_points([midLevelThickness, lowLevelThickness]):
            return "FREEZING RAIN+SLEET"
        elif FRZARainPoly.contains_point([midLevelThickness, lowLevelThickness]):
            return "FREEZING RAIN"
    else:
        return "NONE"

def plotPsblHazType(profileData, ax, precipType):
    if precipType == "NONE":
        suffix = "(No saturated layers)"
        precipcolor = "peru"
    elif precipType == "SNOW":
        suffix = "(See partial thickness)"
        precipcolor = "cornflowerblue"
    elif precipType == "UNKNOWN":
        suffix = "(See partial thickness)"
        precipcolor = "purple"
    elif precipType == "UNKNOWNUNKNOWN":
        precipType = "UNKNOWN"
        suffix = "(Partial thickness\nunavailable)"
        precipcolor = "black"
    elif precipType == "SNOW+RAIN":
        suffix = "(See partial thickness)"
        precipcolor = "mediumturquoise"
    elif precipType == "WINTRY MIX":
        suffix = "(See partial thickness)"
        precipcolor = "mediumslateblue"
    elif precipType == "RAIN":
        suffix = "(See partial thickness)"
        precipcolor = "limegreen"
    elif precipType == "sfcRAIN":
        precipType = "RAIN"
        suffix = "(Based on\n sfc temp)"
        precipcolor = "limegreen"
    elif precipType == "SNOW+SLEET":
        suffix = "(See partial thickness)"
        precipcolor = "slategray"
    elif "FREEZING RAIN" in precipType:
        suffix = "(See partial thickness)"
        precipcolor = "darkorange"
    else:
        suffix = "(See partial thickness)"
        precipcolor = "violet"
    ax.text(0.5, 0.05, "Precip Type:\n"+precipType+"\n"+suffix, color=precipcolor, ha="center", va="bottom", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=10)
    if "HGHT" in profileData.keys():
        if "AGL" not in profileData.keys():
            profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
    if "ukt" in profileData.keys() and "vkt" in profileData.keys():
        uwind = profileData["ukt"].values * units("kt")
        vwind = profileData["vkt"].values * units("kt")
    elif "WSPD" in profileData.keys() and "WDIR" in profileData.keys():
        windspeed = profileData["WSPD"].values * units("kt")
        winddirection = profileData["WDIR"].values * units.deg
        # get u and v wind components from magnitude/direction
        uwind, vwind = mpcalc.wind_components(windspeed, winddirection)
    elif "ums" in profileData.keys() and "vms" in profileData.keys():
        uwind = (profileData["ums"].values * units.meter/units.sec).to(units.kt)
        vwind = (profileData["vms"].values * units.meter/units.sec).to(units.kt)
        profileData["ukt"] = uwind.magnitude
        profileData["vkt"] = vwind.magnitude
    pressure = profileData["LEVEL"].values * units.hPa
    temp = profileData["TEMP"].values * units.degC
    dewpoints = profileData["DWPT"].values * units.degC
    virtT = mpcalc.virtual_temperature_from_dewpoint(pressure, temp, dewpoints)
    height = profileData["AGL"].values * units.m
    sblcl = mpcalc.lcl(pressure[0], virtT[0], dewpoints[0])[0].magnitude
    muparcelP, muparcelT, muparcelTd, _ = mpcalc.most_unstable_parcel(pressure, temp, dewpoints)
    sblcl = profileData.loc[(profileData["LEVEL"] - sblcl).abs().idxmin()]["AGL"] * units.meter
    mlparcel = mpcalc.mixed_parcel(pressure, temp, dewpoints)
    mllcl = mpcalc.lcl(mlparcel[0], mpcalc.virtual_temperature_from_dewpoint(mlparcel[0], mlparcel[1], mlparcel[2]), mlparcel[2])[0].magnitude
    mllcl = profileData.loc[(profileData["LEVEL"] - mllcl).abs().idxmin()]["AGL"] * units.meter
    sbcape, sbcin = mpcalc.surface_based_cape_cin(pressure, temp, dewpoints)
    mlcape, mlcin = mpcalc.mixed_layer_cape_cin(pressure, temp, dewpoints)
    mucape, mucin = mpcalc.most_unstable_cape_cin(pressure, temp, dewpoints)
    llRH = profileData.loc[profileData["LEVEL"] >= profileData.iloc[0]["LEVEL"]-100].mean()["RH"]
    mlRH = profileData.loc[profileData["LEVEL"] >= profileData.iloc[0]["LEVEL"]-350].loc[profileData["LEVEL"] <= profileData.iloc[0]["LEVEL"]-150].mean()["RH"]
    oneKmLapseRate = profileData.iloc[0]["TEMP"] - profileData.loc[(profileData["AGL"] - 1000).abs().idxmin()]["TEMP"]
    sevenToFiveLapse = profileData.loc[(profileData["LEVEL"] - 700).abs().idxmin()]["TEMP"] - profileData.loc[(profileData["LEVEL"] - 500).abs().idxmin()]["TEMP"]
    sevenToFiveLapseRate = sevenToFiveLapse / ((profileData.loc[(profileData["LEVEL"] - 500).abs().idxmin()]["AGL"]-profileData.loc[(profileData["LEVEL"] - 700).abs().idxmin()]["AGL"])/1000)
    freezingLevel = profileData.loc[(profileData["TEMP"]).abs().idxmin()]["AGL"]
    cloudLayer = profileData.loc[profileData["LEVEL"] <= 850].loc[profileData["LEVEL"] >= 300]
    cloudLayerU, cloudLayerV = cloudLayer["ukt"].mean() * units.kt, cloudLayer["vkt"].mean() * units.kt
    LLJ = profileData.loc[(profileData["AGL"] - 1500).abs().idxmin()]
    uLLJ, vLLJ = LLJ["ukt"] * units.kt, LLJ["vkt"] * units.kt
    corfidiUpshearU, corfidiUpshearV = cloudLayerU - uLLJ, cloudLayerV - vLLJ
    corfidiUpshearMag = mpcalc.wind_speed(corfidiUpshearU, corfidiUpshearV)
    pwat = mpcalc.precipitable_water(pressure, dewpoints)
    rm, lm, mw = mpcalc.bunkers_storm_motion(pressure, uwind, vwind, height)
    rmSRH = mpcalc.storm_relative_helicity(profileData["HGHT"].values * units.m, uwind, vwind, depth=3000*units.m, storm_u=rm[0], storm_v=rm[1])[2]
    lmSRH = mpcalc.storm_relative_helicity(profileData["HGHT"].values * units.m, uwind, vwind, depth=3000*units.m, storm_u=lm[0], storm_v=lm[1])[2]
    if rmSRH > lmSRH:
        favored = rm
        favoredSRH = rmSRH
        favored1kmSRH = mpcalc.storm_relative_helicity(profileData["HGHT"].values * units.m, uwind, vwind, depth=1000*units.m, storm_u=rm[0], storm_v=rm[1])[2]
    else:
        favored = lm
        favoredSRH = lmSRH
        favored1kmSRH = mpcalc.storm_relative_helicity(profileData["HGHT"].values * units.m, uwind, vwind, depth=1000*units.m, storm_u=lm[0], storm_v=lm[1])[2]
    uShear, vShear = mpcalc.bulk_shear(pressure, uwind, vwind, height=height, bottom=height[0], depth=6000 * units.meter)
    bulkShear = mpcalc.wind_speed(uShear, vShear)
    uShear8, vShear8 = mpcalc.bulk_shear(pressure, uwind, vwind, height=height, bottom=height[0], depth=8000 * units.meter)
    bulkShear8 = mpcalc.wind_speed(uShear8, vShear8)
    fixedSTP = mpcalc.significant_tornado(sbcape, sblcl, favored1kmSRH, bulkShear).to_base_units().magnitude[0]
    scp = mpcalc.supercell_composite(mucape, favoredSRH, bulkShear).to_base_units().magnitude[0]
    shearms = bulkShear.to(units.meter/units.second).magnitude
    if shearms > 27:
        shearms = 27
    elif shearms < 7:
        shearms = 7.
    mostUnstableMixingRatio = mpcalc.mixing_ratio_from_relative_humidity(muparcelP, muparcelT, mpcalc.relative_humidity_from_dewpoint(muparcelT, muparcelTd)).to("g/kg").magnitude
    if mostUnstableMixingRatio > 13.6:
        mostUnstableMixingRatio = 13.6
    elif mostUnstableMixingRatio < 11:
        mostUnstableMixingRatio = 11
    tempAt500 = profileData.loc[(profileData["LEVEL"] - 500).abs().idxmin()]["TEMP"]
    if tempAt500 > -5.5:
        tempAt500 = -5.5
    ship = -1 * mucape.magnitude * mostUnstableMixingRatio * sevenToFiveLapseRate * tempAt500 * shearms / 42000000
    if mucape < 1300 * units("J/kg"):
        ship = ship*(mucape.magnitude/1300.)
    if sevenToFiveLapseRate < 5.8:
        ship = ship*(sevenToFiveLapseRate/5.8)
    if freezingLevel < 2400:
        ship = ship * (freezingLevel/2400.)
    fourToSix = profileData.loc[profileData["AGL"] <= 6000].loc[profileData["AGL"] >= 4000]
    fourToSixMW = mpcalc.mean_pressure_weighted((fourToSix["LEVEL"].values * units.hPa), fourToSix["ukt"].values * units.knots, fourToSix["vkt"].values * units.knots, height=(fourToSix["AGL"].values * units.meter), bottom=(fourToSix["LEVEL"].values[0] * units.hPa), depth=(fourToSix["LEVEL"].values[0] * units.hPa - fourToSix["LEVEL"].values[-1] * units.hPa))
    fourToSixSRWu, fourToSixSRWv = (fourToSixMW[0]-favored[0]), (fourToSixMW[1]-favored[1])
    fourToSixSRW = mpcalc.wind_speed(fourToSixSRWu, fourToSixSRWv)
    if mlcin > -50 * units("J/kg"):
        cinTerm = 1
    elif mlcin < -200 * units("J/kg"):
        cinTerm = 0
    else:
        cinTerm = ((mlcin.magnitude + 200.) / 150.)
    eSTP = (fixedSTP * cinTerm)
    if eSTP >= 3 and fixedSTP >= 3 and favored1kmSRH >= 200 * units.meter**2 / units.second**2 and favoredSRH >= 200 * units.meter**2 / units.second**2 and fourToSixSRW >= 15 and bulkShear8.magnitude > 45 and sblcl.magnitude < 1000 and mllcl < 1200 and oneKmLapseRate >= 5 and mlcin > -50 and sbcape == mucape:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nPDS TOR", color="magenta", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if eSTP >= 3 or fixedSTP >= 4:
        if mlcin >= -125 * units("J/kg"):
            if sbcape == mucape:
                ax.text(0.5, 0.9, "Possible\nHazard Type:\nTOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
                return
    if eSTP >= 1 or fixedSTP >= 1:
        if fourToSixSRW >= 15 * units.kt or bulkShear8 > 40 * units.kt:
            if mlcin >= -50 * units("J/kg") and sbcape == mucape:
                ax.text(0.5, 0.9, "Possible\nHazard Type:\nTOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
                return
    if eSTP >= 1 or fixedSTP >= 1:
        if np.mean(llRH, mlRH) >= 0.6 and oneKmLapseRate >= 5 and mlcin > -50*units("J/kg") and sbcape == mucape:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nTOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if eSTP >= 1 or fixedSTP >= 1:
        if mlcin > -150 * units("J/kg") and sbcape == mucape:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL TOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if mlcin > -50 *units("J/kg") and mucape == sbcape:
        if eSTP >= 1 and favoredSRH >= 150 * units.meter**2 / units.second**2:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL TOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
        if fixedSTP >= 0.5 and favored1kmSRH >= 150 * units.meter**2 / units.second**2:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL TOR", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if fixedSTP >= 1 or scp >= 4 or eSTP >= 1:
        if mucin > -50 * units("J/kg"):
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nSVR", color="yellow", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if mucin > -50 *units("J/kg"):
        if ship >=1 and scp >= 2:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nSVR", color="yellow", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
        # if dcape >= 750 * units("J/kg"):
        #     ax.text(0.5, 0.9, "Possible\nHazard Type:\nSVR", color="yellow", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        #     return
    if mucin > -75 * units("J/kg"):
        if ship >= 0.5 or scp >= 0.5:
            ax.text(0.5, 0.9, "Possible\nHazard Type:\nMRGL SVR", color="dodgerblue", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
            return
    if pwat >= 2 * units("in") and corfidiUpshearMag <= 25 * units("kt"):
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nFLASH FLOOD", color="forestgreen", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if mpcalc.wind_speed(uwind[0], vwind[0]) > 35 * units("mph") and "SNOW" in precipType:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nBLIZZARD", color="darkblue", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if mpcalc.apparent_temperature(temp[0], mpcalc.relative_humidity_from_dewpoint(temp[0], dewpoints[0]), mpcalc.wind_speed(uwind[0], vwind[0]), mask_undefined=False) > 105 * units.degF:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nEXCESSIVE HEAT", color="red", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    if mpcalc.apparent_temperature(temp[0], mpcalc.relative_humidity_from_dewpoint(temp[0], dewpoints[0]), mpcalc.wind_speed(uwind[0], vwind[0]), mask_undefined=False) < -20 * units.degF:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nWIND CHILL", color="blue", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return
    else:
        ax.text(0.5, 0.9, "Possible\nHazard Type:\nNONE", color="black", ha="center", va="top", transform=ax.transAxes, clip_on=False, zorder=5, fontsize=12)
        return

def plotHodograph(profileData, ax):
    hodoPlot = plots.Hodograph(ax, component_range=200)
    hodoPlot.add_grid(increment=10)
    if "ukt" in profileData.keys() and "vkt" in profileData.keys():
        uwind = profileData["ukt"].values * units("kt")
        vwind = profileData["vkt"].values * units("kt")
    elif "WSPD" in profileData.keys() and "WDIR" in profileData.keys():
        windspeed = profileData["WSPD"].values * units("kt")
        winddirection = profileData["WDIR"].values * units.deg
        # get u and v wind components from magnitude/direction
        uwind, vwind = mpcalc.wind_components(windspeed, winddirection)
    elif "ums" in profileData.keys() and "vms" in profileData.keys():
        uwind = (profileData["ums"].values * units.meter/units.sec).to(units.kt)
        vwind = (profileData["vms"].values * units.meter/units.sec).to(units.kt)
        profileData["ukt"] = uwind.magnitude
        profileData["vkt"] = vwind.magnitude
    [ax.text(0, i, f"{i} kt", color="gray", clip_on=True, zorder=6) for i in np.arange(-200, 201, 10).astype(int)]
    [ax.text(i, 0, f"{i} kt", color="gray", clip_on=True, zorder=6) for i in np.arange(-200, 201, 10).astype(int)]

    if "HGHT" in profileData.keys():
        if "AGL" not in profileData.keys():
            profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
        profileDataWithWind = profileData.loc[~np.isnan(profileData["ukt"])].reset_index(drop=True)
        uwind = profileDataWithWind["ukt"].values * units("kt")
        vwind = profileDataWithWind["vkt"].values * units("kt")
        hodoPlot.plot_colormapped(uwind, vwind, profileDataWithWind["AGL"].values * units("m"), colors=["fuchsia", "firebrick", "limegreen", "gold", "darkturquoise", "darkturquoise"], intervals=np.array([0, 1, 3, 6, 9, 12, 15])*units("km"))
        maxKm = np.nanmax(profileDataWithWind["AGL"].values) // 1000
        kmToLabel = list(np.arange(1, np.min([maxKm+1, 7])))
        ax.scatter(uwind[0], vwind[0], color="black", marker="o", s=125, zorder=4)
        ax.scatter(uwind[0], vwind[0], color="white", marker="o", s=100, zorder=4)
        ax.annotate("0", [uwind[0], vwind[0]], color="black", clip_on=True, zorder=5, ha="center", va="center")
        if maxKm >= 1:
            kmToLabel.extend([0.5])
            if maxKm >=9:
                kmToLabel.extend([9])
                if maxKm >= 12:
                    kmToLabel.extend([12])
                    if maxKm >= 15:
                        kmToLabel.extend([15])
        for i in kmToLabel:
            closestValueToTarget = profileDataWithWind.loc[(profileDataWithWind["AGL"] - i*1000).abs().idxmin()]
            selectedIdx = closestValueToTarget.name
            if closestValueToTarget["AGL"] > i*1000:
                secondClosestValueToTarget = profileDataWithWind.iloc[selectedIdx-1]
            else:
                secondClosestValueToTarget = profileDataWithWind.iloc[selectedIdx+1]
            targetIsh = pd.concat([closestValueToTarget, secondClosestValueToTarget], axis=1).mean(axis=1)
            utarget, vtarget = mpcalc.wind_components(targetIsh["WSPD"] * units("kt"), targetIsh["WDIR"] * units.deg)
            ax.scatter(utarget, vtarget, color="black", marker="o", s=125, zorder=4)
            ax.scatter(utarget, vtarget, color="white", marker="o", s=100, zorder=4)
            textToPlot = str(int(i)) if i != 0.5 else ".5"
            ax.annotate(textToPlot, [utarget, vtarget], color="black", clip_on=True, zorder=5, ha="center", va="center")
        rm, lm, mw = mpcalc.bunkers_storm_motion(profileDataWithWind["LEVEL"].values * units.hPa, uwind, vwind, (profileDataWithWind["HGHT"].values * units.m))
        rmMag, rmDir = mpcalc.wind_speed(*rm), mpcalc.wind_direction(*rm)
        lmMag, lmDir = mpcalc.wind_speed(*lm), mpcalc.wind_direction(*lm)
        mwMag, mwDir = mpcalc.wind_speed(*mw), mpcalc.wind_direction(*mw)
        rmSRH = mpcalc.storm_relative_helicity(profileDataWithWind["HGHT"].values * units.m, uwind, vwind, depth=3000*units.m, storm_u=rm[0], storm_v=rm[1])[2]
        lmSRH = mpcalc.storm_relative_helicity(profileDataWithWind["HGHT"].values * units.m, uwind, vwind, depth=3000*units.m, storm_u=lm[0], storm_v=lm[1])[2]
        lowest500m = profileDataWithWind.loc[profileDataWithWind["AGL"] <= 500].mean()
        lowest500u, lowest500v = lowest500m["ukt"] * units.kt, lowest500m["vkt"] * units.kt
        if rmSRH.magnitude >= lmSRH.magnitude:
            ax.scatter(rm[0], rm[1], color="white", edgecolor="red", marker="o", s=50, zorder=4, label=f"Bunkers Right [{rmMag.magnitude:.1f} kt, {rmDir.magnitude:.1f}°]")
            ax.scatter(lm[0], lm[1], color="white", edgecolor="blue", marker="o", s=50, zorder=4, alpha=0.5, label=f"Bunkers Left [{lmMag.magnitude:.1f} kt, {lmDir.magnitude:.1f}°]")
            favoredU, favoredV = rm[0], rm[1]
        else:
            ax.scatter(rm[0], rm[1], color="white", edgecolor="red", marker="o", s=50, zorder=4, alpha=0.5, label=f"Bunkers Right [{rmMag.magnitude:.1f} kt, {rmDir.magnitude:.1f}°]")
            ax.scatter(lm[0], lm[1], color="white", edgecolor="blue", marker="o", s=50, zorder=4, label=f"Bunkers Left [{lmMag.magnitude:.1f} kt, {lmDir.magnitude:.1f}°]")
            favoredU, favoredV = rm[0], rm[1]
        dtmU = np.mean([favoredU.magnitude, lowest500u.magnitude]) * units.kt
        dtmV = np.mean([favoredV.magnitude, lowest500v.magnitude]) * units.kt
        dtmMag, dtmDir = mpcalc.wind_speed(dtmU, dtmV), mpcalc.wind_direction(dtmU, dtmV)
        ax.scatter(dtmU.magnitude, dtmV.magnitude, color="white", edgecolor="darkviolet", marker="v", s=50, zorder=4, label=f"Deviant Tornado Motion [{dtmMag.magnitude:.1f} kt, {dtmDir.magnitude:.1f}°]")
        ax.scatter(mw[0], mw[1], color="white", edgecolor="sienna", marker="s", s=50, zorder=4, label=f"0->6km Mean Wind [{mwMag.magnitude:.1f} kt, {mwDir.magnitude:.1f}°]")
        cloudLayer = profileDataWithWind.loc[profileDataWithWind["LEVEL"] <= 850].loc[profileDataWithWind["LEVEL"] >= 300]
        cloudLayerU, cloudLayerV = cloudLayer["ukt"].mean() * units.kt, cloudLayer["vkt"].mean() * units.kt
        LLJ = profileDataWithWind.loc[(profileDataWithWind["AGL"] - 1500).abs().idxmin()]
        uLLJ, vLLJ = LLJ["ukt"] * units.kt, LLJ["vkt"] * units.kt
        corfidiUpshearU, corfidiUpshearV = cloudLayerU - uLLJ, cloudLayerV - vLLJ
        corfidiUpshearMag, corfidiUpshearDir = mpcalc.wind_speed(corfidiUpshearU, corfidiUpshearV), mpcalc.wind_direction(corfidiUpshearU, corfidiUpshearV)
        ax.scatter(corfidiUpshearU.magnitude, corfidiUpshearV.magnitude, color="limegreen", marker="2", s=50, zorder=4, label=f"Corfidi Upshear [{corfidiUpshearMag.magnitude:.1f} kt, {corfidiUpshearDir.magnitude:.1f}°]")
        corfidiDownshearU, corfidiDownshearV = cloudLayerU + corfidiUpshearU, cloudLayerV + corfidiUpshearV
        corfidiDownshearMag, corfidiDownshearDir = mpcalc.wind_speed(corfidiDownshearU, corfidiDownshearV), mpcalc.wind_direction(corfidiDownshearU, corfidiDownshearV)
        ax.scatter(corfidiDownshearU.magnitude, corfidiDownshearV.magnitude, color="darkgreen", marker="1", s=50, zorder=4, label=f"Corfidi Downshear [{corfidiDownshearMag.magnitude:.1f} kt, {corfidiDownshearDir.magnitude:.1f}°]")
        if "inEIL" in profileDataWithWind.keys():
            eil = profileDataWithWind.loc[profileDataWithWind["inEIL"] == 1]
            critAngle = mpcalc.critical_angle(profileDataWithWind["LEVEL"].values * units.hPa, uwind, vwind, profileDataWithWind["HGHT"].values * units.m, favoredU, favoredV)
            ax.plot([favoredU.magnitude, eil["ukt"].iloc[0]], [favoredV.magnitude, eil["vkt"].iloc[0]], color="teal", linestyle="--", linewidth=0.5, zorder=1, label="Eff. Inflow [Crit. Angle: {:.1f}°]".format(critAngle.magnitude))
            ax.plot([favoredU.magnitude, eil["ukt"].iloc[-1]], [favoredV.magnitude, eil["vkt"].iloc[-1]], color="teal", linestyle="--", linewidth=0.5, zorder=1)
            
            closestValueToTarget = profileDataWithWind.loc[(profileDataWithWind["AGL"] - 500).abs().idxmin()]
            selectedIdx = closestValueToTarget.name
            if closestValueToTarget["AGL"] > 500:
                secondClosestValueToTarget = profileDataWithWind.iloc[selectedIdx-1]
            else:
                secondClosestValueToTarget = profileDataWithWind.iloc[selectedIdx+1]
            targetIsh = pd.concat([closestValueToTarget, secondClosestValueToTarget], axis=1).mean(axis=1)
            utarget, vtarget = mpcalc.wind_components(targetIsh["WSPD"] * units("kt"), targetIsh["WDIR"] * units.deg)

            ax.plot([utarget.magnitude, eil["ukt"].iloc[0]], [vtarget.magnitude, eil["vkt"].iloc[0]], color="mediumvioletred", linestyle="--", linewidth=0.5, zorder=1)
        ax.legend()
    else:
        hodoPlot.plot_colormapped(uwind, vwind, profileData["LEVEL"].values * units.hPa, cmap="plasma")
    ax.set_xlim(np.nanmin([np.nanmin(profileData["ukt"].loc[profileData["AGL"] <= 12000]), corfidiUpshearU.magnitude, corfidiDownshearU.magnitude, dtmU.magnitude, mw[0].magnitude, rm[0].magnitude, lm[0].magnitude])-2.5, np.nanmax([np.nanmax(profileData["ukt"].loc[profileData["AGL"] <= 12000]), corfidiUpshearU.magnitude, corfidiDownshearU.magnitude, dtmU.magnitude, mw[0].magnitude, rm[0].magnitude, lm[0].magnitude])+2.5)
    ax.set_ylim(np.nanmin([np.nanmin(profileData["vkt"].loc[profileData["AGL"] <= 12000]), corfidiUpshearV.magnitude, corfidiDownshearV.magnitude, dtmV.magnitude, mw[1].magnitude, rm[1].magnitude, lm[1].magnitude])-2.5, np.nanmax([np.nanmax(profileData["vkt"].loc[profileData["AGL"] <= 12000]), corfidiUpshearV.magnitude, corfidiDownshearV.magnitude, dtmV.magnitude, mw[1].magnitude, rm[1].magnitude, lm[1].magnitude])+2.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

def plotThermalWind(profileData, ax, latitude):
    if latitude is not None:
        if "ums" in profileData.keys() and "vms" in profileData.keys():
            pass
        elif "WSPD" in profileData.keys() and "WDIR" in profileData.keys():
            windspeed = profileData["WSPD"].values * units("kt")
            winddirection = profileData["WDIR"].values * units.deg
            # get u and v wind components from magnitude/direction
            uwind, vwind = mpcalc.wind_components(windspeed, winddirection)
            profileData["ums"] = uwind.to(units.meter/units.sec)
            profileData["vms"] = vwind.to(units.meter/units.sec)
        elif "ukt" in profileData.keys() and "vkt" in profileData.keys():
            profileData["ums"] = (profileData["ukt"].values * units.kt).to(units.meter/units.sec)
            profileData["vms"] = (profileData["vkt"].values * units.kt).to(units.meter/units.sec)
        profileWithWinds = profileData.loc[~np.isnan(profileData["ums"])].reset_index(drop=True)
        uflip = np.flip((profileWithWinds["ums"].values * units.meter/units.sec))
        uflip, uunits = uflip.magnitude, uflip.units
        vflip = np.flip((profileWithWinds["vms"].values * units.meter/units.sec))
        vflip, vunits = vflip.magnitude, vflip.units
        pflip = np.flip((profileWithWinds["LEVEL"].values * units.hPa)).to(units.pascal)
        pflip, punits = pflip.magnitude, pflip.units
        dvdP = np.flip(mpcalc.first_derivative(vflip * vunits, x=pflip * punits))
        profileWithWinds["dvdP"], dvdPunits = dvdP.magnitude, dvdP.units
        dudP = np.flip(mpcalc.first_derivative(uflip * uunits, x=pflip * punits))
        profileWithWinds["dudP"], dudPunits = dudP.magnitude, dudP.units

        f = mpcalc.coriolis_parameter(latitude * units.deg)
        R = constants.dry_air_gas_constant.to_base_units()
        
        dTdx = - (f/R) * (profileWithWinds["LEVEL"].values * units.hPa).to(units.pascal) * dvdP
        dTdy = (f/R) * (profileWithWinds["LEVEL"].values * units.hPa).to(units.pascal) * dudP
        dTdt = (- (profileWithWinds["ums"].values * units.meter/units.sec) * dTdx - (profileWithWinds["vms"].values * units.meter/units.sec) * dTdy).to(units.kelvin / units.hour)
        profileWithWinds["dTdt"] = dTdt.magnitude
        ax.fill_betweenx(profileWithWinds["LEVEL"].values * units("hPa"), profileWithWinds["dTdt"].values * units.kelvin/units.hour, 0, where=profileWithWinds["dTdt"].values > 0, color="red", alpha=0.5, zorder=2)
        ax.fill_betweenx(profileWithWinds["LEVEL"].values * units("hPa"), profileWithWinds["dTdt"].values * units.kelvin/units.hour, 0, where=profileWithWinds["dTdt"].values < 0, color="blue", alpha=0.5, zorder=2)

        minimas = (np.diff(np.sign(np.diff(dTdt))) > 0).nonzero()[0] + 1 
        maximas = (np.diff(np.sign(np.diff(dTdt))) < 0).nonzero()[0] + 1
        minmaxes = np.sort(np.append(minimas, maximas))
        lastMaxText = None
        lastMaxLvl = np.nanmax(profileData["LEVEL"].values)+100
        lastMaxValue = 0
        lastMinText = None
        lastMinLvl = np.nanmax(profileData["LEVEL"].values)+100
        lastMinValue = 0
        for minOrMax in minmaxes:
            value = dTdt[minOrMax]
            lvl = profileData.iloc[minOrMax]["LEVEL"]
            if lvl < 100:
                break
            if value < 0:
                if lastMinText is not None:
                    if np.abs(lvl - lastMinLvl) < 20:
                        if np.abs(value) > np.abs(lastMinValue):
                            lastMinText.remove()
                        else:
                            continue
                lastMinText = ax.text(np.nanmin(dTdt), lvl, f"{value.magnitude:.1f}", color="blue",  ha="left", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], clip_on=False, alpha=0.5, zorder=3)
                lastMinLvl = lvl
                lastMinValue = value
            else:
                if lastMaxText is not None:
                    if np.abs(lvl - lastMaxLvl) < 20:
                        if np.abs(value) > np.abs(lastMaxValue):
                            lastMaxText.remove()
                        else:
                            continue
                lastMaxText = ax.text(np.nanmax(dTdt), lvl, f"{value.magnitude:.1f}", color="red",  ha="right", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], clip_on=False, alpha=0.5, zorder=3)
                lastMaxLvl = lvl
                lastMaxValue = value
        ax.set_xlabel("Temp. Adv.\n(K/hr)") # that's really jank
        ax.set_xlim(np.nanmin(dTdt), np.nanmax(dTdt))
    else:
        ax.set_xlabel("RH %")
        ax.set_xlim(0, 100)
    for top in range(100, 1101, 100):
        bottom = top + 100
        selectedData = profileData.loc[(profileData["LEVEL"] <= bottom) & (profileData["LEVEL"] > top)]
        if len(selectedData) == 0:
            continue
        RHinLayer = mpcalc.mean_pressure_weighted(selectedData["LEVEL"].values * units.hPa, selectedData["RH"].values, height=(selectedData["AGL"].values*units.meter), bottom=(selectedData["LEVEL"].values[0]*units.hPa), depth=(selectedData["LEVEL"].values[0]*units.hPa - selectedData["LEVEL"].values[-1]*units.hPa))[0].magnitude
        ax.plot([RHinLayer, RHinLayer], [top, bottom], color="forestgreen", linewidth=1, zorder=1, transform=ax.get_yaxis_transform())
        ax.fill_betweenx([bottom, top], RHinLayer, 0, color="limegreen", alpha=0.5, zorder=1, transform=ax.get_yaxis_transform())
        ax.text(RHinLayer, (top+bottom)/2, f"{int(RHinLayer*100)}%", color="forestgreen",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], clip_on=False, alpha=0.5, zorder=3, transform=ax.get_yaxis_transform())
    ax.hlines(range(100, 1001, 100), xmin=-100, xmax=100, color="gray", linewidth=0.5, zorder=1)
    ax.tick_params(which="both", labelleft=False, direction="in", pad=-9)
    ax.set_ylabel("")
    return profileData

def plotSkewT(profileData, skew, parcelType="in"):
    pressure = profileData["LEVEL"].values * units("hPa")
    temperature = profileData["TEMP"].values * units("degC")
    dewpoints = profileData["DWPT"].values * units("degC")
    profileData["RH"] = mpcalc.relative_humidity_from_dewpoint(temperature, dewpoints)
    virtT = mpcalc.virtual_temperature_from_dewpoint(pressure, temperature, dewpoints)
    wetbulb = mpcalc.wet_bulb_temperature(pressure, temperature, dewpoints)
    # Plot data
    skew.plot(pressure, temperature, "red", zorder=4)
    skew.plot(pressure, dewpoints, "lime", zorder=5)
    skew.plot(pressure, virtT, "red", linestyle=":", zorder=4)
    skew.plot(pressure, wetbulb, "cyan", linewidth=0.5, zorder=3)
    
    if "WSPD" in profileData.keys() and "WDIR" in profileData.keys():
        windspeed = profileData["WSPD"].values * units("kt")
        winddirection = profileData["WDIR"].values * units.deg
        # get u and v wind components from magnitude/direction
        profileData["ukt"], profileData["vkt"] = mpcalc.wind_components(windspeed, winddirection)
        profileWithWind = profileData.loc[~np.isnan(profileData["ukt"])].reset_index(drop=True)
        # we don't need to plot every packet of wind data, so space it out a little
        mask = mpcalc.resample_nn_1d(profileWithWind["LEVEL"],  np.logspace(4, 2))
        skew.plot_barbs(profileWithWind["LEVEL"][mask] * units.hPa, profileWithWind["ukt"][mask] * units.kt, profileWithWind["vkt"][mask] * units.kt)
    skew.ax.set_xlim(10*(np.nanmin(temperature.magnitude) // 10)+10, (10*np.nanmax(temperature.magnitude) // 10)+10)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.set_ylim(np.nanmax(pressure.magnitude)+5, 100)
    skew.ax.tick_params(which="both", direction="in")
    skew.ax.tick_params(axis="x", direction="in", pad=-9)
    skew.plot_dry_adiabats(color="gray", linewidths=0.2, zorder=1)
    skew.plot_moist_adiabats(color="gray", linewidths=0.2, zorder=1)
    skew.ax.vlines([-12, -17], np.nanmin(profileData["LEVEL"].values), np.nanmax(profileData["LEVEL"].values), color="blue", linestyle="--", linewidth=0.5, zorder=1)

    skew.ax.text(wetbulb[0], -0.012, f"{int((wetbulb[0]).to(units.degF).magnitude)} °F", color="cyan",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_xaxis_transform())
    skew.ax.text(profileData['DWPT'].iloc[0]-5, -0.012, f"{str(int((profileData['DWPT'].iloc[0] * units.degC).to(units.degF).magnitude))} °F", color="lime",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_xaxis_transform(), alpha=0.5)
    skew.ax.text(profileData['TEMP'].iloc[0]+5, -0.012, f"{int((profileData['TEMP'].iloc[0] * units.degC).to(units.degF).magnitude)} °F", color="red",  ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_xaxis_transform(), alpha=0.5)

    if "HGHT" in profileData.keys():
        if "AGL" not in profileData.keys():
            profileData["AGL"] = profileData["HGHT"] - profileData["HGHT"].iloc[0]
        kmToInterp = np.arange(0, 15001, 1000)
        interpPressures = np.interp(kmToInterp, profileData["AGL"].values, profileData["LEVEL"].values)
        colorsAndKilometers = {
                (0, 1): "fuchsia",
                (1, 3): "firebrick",
                (3, 6): "limegreen",
                (6, 9): "gold",
                (9, 12): "darkturquoise",
                (12, 15): "darkturquoise"
        }
        for indices, color in reversed(colorsAndKilometers.items()):
            bot, top = indices
            skew.ax.add_patch(Polygon([(0, interpPressures[bot]), (0.01, interpPressures[bot]), (0.01, interpPressures[top]), (0, interpPressures[top])], closed=True, color=color, alpha=0.5, zorder=1, transform=skew.ax.get_yaxis_transform()))
        for i in [0, 1, 2, 3, 4, 5, 6, 9, 12, 15]:
            if i == 0:
                skew.ax.text(0.01, interpPressures[i]-5, f"SFC: {interpPressures[i]:.1f} hPa", color="black", transform=skew.ax.get_yaxis_transform())
            else:
                skew.ax.text(0.01, interpPressures[i], f"{str(int(i))} km: {interpPressures[i]:.1f} hPa", color="black", transform=skew.ax.get_yaxis_transform())
                skew.ax.plot([0, 0.05], [interpPressures[i], interpPressures[i]], color="gray", linewidth=.75, transform=skew.ax.get_yaxis_transform())
    inflowBottom = 0 * units.hPa
    inflowTop = 0 * units.hPa
    inEILArr = np.zeros(len(profileData))
    for i in range(len(profileData)):
        slicedProfile = profileData.iloc[i:]
        capeProfile = mpcalc.parcel_profile(slicedProfile["LEVEL"].values * units.hPa, slicedProfile["TEMP"].values[0] * units.degC, slicedProfile["DWPT"].values[0] * units.degC)
        cape, cinh = mpcalc.cape_cin(slicedProfile["LEVEL"].values * units("hPa"), slicedProfile["TEMP"].values * units.degC, slicedProfile["DWPT"].values * units.degC, parcel_profile=capeProfile)
        if cape.magnitude >= 100 and cinh.magnitude >= -250:
                inEILArr[i] = 1
                inflowTop = profileData["LEVEL"].iloc[i]  * units.hPa
                if inflowBottom.magnitude == 0:
                    inflowBottom = profileData["LEVEL"].iloc[i] * units.hPa
        else:
            if inflowBottom.magnitude != 0:
                break
    profileData["inEIL"] = inEILArr
    eilData = profileData.loc[profileData["inEIL"] == 1]
    eilRH = int(eilData["RH"].mean() * 100)
    if "AGL" in profileData.keys():
        aglInflowTop = eilData["AGL"].values[-1]
        aglInflowBot = eilData["AGL"].values[0]
        skew.ax.text(0.16, inflowTop.magnitude, f"Effective Inflow Layer\n{inflowBottom.magnitude:.1f} - {inflowTop.magnitude:.1f} hPa\nAGL: {int(aglInflowBot)} - {int(aglInflowTop)} m\nRH: {eilRH}%", color="teal",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)
    else:
        skew.ax.text(0.16, inflowTop.magnitude, f"Effective Inflow Layer\n{inflowBottom.magnitude:.1f} - {inflowTop.magnitude:.1f}\nRH: {eilRH}%", color="teal",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)
    skew.ax.plot([0.15, 0.15], [inflowBottom.magnitude, inflowTop.magnitude], color="teal", linewidth=1.25, transform=skew.ax.get_yaxis_transform())
    skew.ax.plot([0, 0.25], [inflowBottom.magnitude, inflowBottom.magnitude], color="teal", linewidth=1.25, transform=skew.ax.get_yaxis_transform())
    skew.ax.plot([0, 0.25], [inflowTop.magnitude, inflowTop.magnitude], color="teal", linewidth=1.25, transform=skew.ax.get_yaxis_transform())

    if parcelType == "sb":
        parcel = mpcalc.parcel_profile(pressure, virtT[0], dewpoints[0])
        lcl = mpcalc.lcl(pressure[0], virtT[0], dewpoints[0])
        initIdx = 0
    elif parcelType == "mu":
        _, initTemp, initDewpoint, initIdx = mpcalc.most_unstable_parcel(pressure, temperature, dewpoints)
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(pressure[initIdx:][0], initTemp, initDewpoint)
        parcel = mpcalc.parcel_profile(pressure[initIdx:], initVirtT, initDewpoint)
        lcl = mpcalc.lcl(pressure[initIdx:][0], initVirtT, initDewpoint)
    elif parcelType == "ml":
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(pressure, temperature, dewpoints)
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
        parcel = mpcalc.parcel_profile(pressure, initVirtT, initDewpoint)
        lcl = mpcalc.lcl(pressure[0], initVirtT, initDewpoint)
        initIdx = 0
    elif parcelType == "in":
        initPressure, initTemp, initDewpoint = mpcalc.mixed_parcel(pressure, temperature, dewpoints, parcel_start_pressure=inflowBottom, depth=(inflowBottom - inflowTop))
        initVirtT = mpcalc.virtual_temperature_from_dewpoint(initPressure, initTemp, initDewpoint)
        parcel = mpcalc.parcel_profile(pressure, initVirtT, initDewpoint)
        lcl = mpcalc.lcl(pressure[0], initVirtT, initDewpoint)
        initIdx = profileData.loc[profileData["LEVEL"] <= inflowBottom.magnitude].index[0]
    skew.plot(pressure[initIdx:], parcel, "black", linewidth=1, zorder=6, linestyle="dashdot")
    skew.ax.plot([0, .95], [lcl[0].magnitude, lcl[0].magnitude], color="mediumseagreen", linewidth=1, transform=skew.ax.get_yaxis_transform())
    skew.ax.text(0.875, lcl[0].magnitude, f"LCL: {lcl[0].magnitude:.1f} hPa", color="mediumseagreen",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
    lfc = mpcalc.lfc(pressure[initIdx:], virtT[initIdx:], dewpoints[initIdx:], parcel_temperature_profile=parcel)
    skew.ax.plot([0, .95], [lfc[0].magnitude, lfc[0].magnitude], color="goldenrod", linewidth=1, transform=skew.ax.get_yaxis_transform())
    skew.ax.text(0.875, lfc[0].magnitude, f"LFC: {lfc[0].magnitude:.1f} hPa", color="goldenrod",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
    el = mpcalc.el(pressure[initIdx:], virtT[initIdx:], dewpoints[initIdx:], parcel_temperature_profile=parcel)
    skew.ax.plot([0, .95], [el[0].magnitude, el[0].magnitude], color="mediumpurple", linewidth=1, transform=skew.ax.get_yaxis_transform())
    skew.ax.text(0.875, el[0].magnitude, f"EL: {el[0].magnitude:.1f} hPa", color="mediumpurple",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())

    cloudLayer = profileData.loc[profileData["LEVEL"] <= lcl[0].magnitude].loc[profileData["LEVEL"] >= el[0].magnitude]
    inCloud = np.zeros(len(profileData))
    inCloud[cloudLayer.index] = 1
    profileData["inCloud"] = inCloud
   
    listOfDGZs = []
    thisDGZ = []
    for i in range(len(profileData)):
        thisTemp = profileData["TEMP"].iloc[i]
        if thisTemp <= -12:
            if thisTemp >= -17:
                if len(thisDGZ) == 0:
                    thisDGZ.append(i)
            else:
                if len(thisDGZ) > 0:
                    thisDGZ.append(i)
                    listOfDGZs.append(thisDGZ)
                    thisDGZ = []
        else:
            if len(thisDGZ) > 0:
                thisDGZ.append(i)
                listOfDGZs.append(thisDGZ)
                thisDGZ = []
    for dgz in listOfDGZs:
        dgzData = profileData.iloc[slice(dgz[0], dgz[1])]
        skew.shade_area(dgzData["LEVEL"].values * units("hPa"), dgzData["TEMP"].values * units.degC, dgzData["DWPT"].values * units.degC, color="blue", alpha=0.2, zorder=6)
        skew.ax.plot([0, 1], [dgzData["LEVEL"].iloc[-1], dgzData["LEVEL"].iloc[-1]], color="blue", linewidth=.75, transform=skew.ax.get_yaxis_transform())
        skew.ax.plot([0, 1], [dgzData["LEVEL"].iloc[0], dgzData["LEVEL"].iloc[0]], color="blue", linewidth=.75, transform=skew.ax.get_yaxis_transform())
        if "AGL" in profileData.keys():
            skew.ax.text(0.15, dgzData.mean()["LEVEL"], f"Dendritic Growth Zone\n{dgzData['LEVEL'].iloc[0]:.1f} - {dgzData['LEVEL'].iloc[-1]:.1f}hPa\nAGL: {int(dgzData['AGL'].iloc[0])} - {int(dgzData['AGL'].iloc[-1])}m\nRH: {int(dgzData.mean()['RH']*100)}%", color="blue",  ha="left", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)
        else:
            skew.ax.text(0.15, dgzData.mean()["LEVEL"], f"Dendritic Growth Zone\n{dgzData['LEVEL'].iloc[0]:.1f} - {dgzData['LEVEL'].iloc[-1]:.1f}hPa\nRH: {int(dgzData.mean()['RH']*100)}%", color="blue",  ha="left", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)
    return profileData
    


def plotSounding(profileData, outputPath, icao, time):
    latitude, longitude = None, None
    airports = pd.read_csv(get_test_data("airport-codes.csv"))
    thisAirport = airports.loc[airports["ident"] == icao]
    if len(thisAirport) == 0:
        thisAirport = airports.loc[airports["iata_code"] == icao]
    if len(thisAirport) == 0:
        thisAirport = airports.loc[airports["local_code"] == icao]
    if len(thisAirport) == 0:
        thisAirport = airports.loc[airports["gps_code"] == icao]
    if len(thisAirport) == 1:
        icao = thisAirport["ident"].values[0]
        latitude = thisAirport["latitude_deg"].values[0]
        longitude = thisAirport["longitude_deg"].values[0]
    else:
        for urlToFetch in ["https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/gfs.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/hrrr.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/hires_conus.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/hires-ak.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/nam.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/nam3km.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/rap.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/sharp.csv",
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/spc_ua.csv"
                           "https://raw.githubusercontent.com/sharppy/SHARPpy/main/datasources/sref.csv"
                           ]:
            sharppyLocs = pd.read_csv(urlToFetch)
            thisAirport = sharppyLocs.loc[sharppyLocs["icao"] == icao]
            if len(thisAirport) == 0:
                thisAirport = sharppyLocs.loc[sharppyLocs["iata"] == icao]
            if len(thisAirport) == 1:
                icao = thisAirport["icao"].values[0]
                latitude = thisAirport["lat"].values[0]
                longitude = thisAirport["lon"].values[0]
                break
    if len(thisAirport) == 0:
        print(f"Unable to determine location of {icao}. Thermal wind and mapping are unavailable.")


    fig = plt.figure()
    
    tax = fig.add_axes([1/20, 14/16, 18/20, 1/16])
    if latitude is not None and longitude is not None:
        tax.text(0.5, 0.5, f"Observed Sounding -- {time.strftime('%H:%M UTC %d %b %Y')} -- {icao} ({latitude:.2f}, {longitude:.2f})", ha="center", va="center", transform=tax.transAxes)
    else:
        tax.text(0.5, 0.5, f"Observed Sounding -- {time.strftime('%H:%M UTC %d %b %Y')} -- {icao}", ha="center", va="center", transform=tax.transAxes)
    tax.axis("off")
    skew = plots.SkewT(fig, rect=[1/20, 4/16, 10/20, 10/16])
    profileData = plotSkewT(profileData, skew)
    skew.ax.patch.set_alpha(0)

    thermalWindAx = fig.add_axes([11/20, 4/16, 1/20, 10/16])
    thermalWindAx.set_yscale("log")
    thermalWindAx.set_ylim(skew.ax.get_ylim())
    if latitude is not None:
        thermalWindAx.text(0.5, 0.95, "Thermal Wind\nRel. Humidity", ha="center", va="center", fontsize=9, transform=thermalWindAx.transAxes)
    else:
        thermalWindAx.text(0.5, 0.95, "Rel. Humidity", ha="center", va="center", fontsize=9, transform=thermalWindAx.transAxes)
    profileData = plotThermalWind(profileData, thermalWindAx, latitude)
    thermalWindAx.patch.set_alpha(0)
    
    hodoAx = fig.add_axes([12/20, 9/16, 7/20, 5/16])
    plotHodograph(profileData, hodoAx)
    hodoAx.patch.set_alpha(0)
    
    partialThicknessAx = fig.add_axes([14/20, 7/16, 2/20, 2/16])
    precipType = plotPartialThickness(profileData, partialThicknessAx)
    
    psblHazTypeAx = fig.add_axes([12/20, 7/16, 2/20, 2/16])
    psblHazTypeAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotPsblHazType(profileData, psblHazTypeAx, precipType)



    mapAx = fig.add_axes([16/20, 7/16, 3/20, 2/16], projection=ccrs.PlateCarree())
    if latitude is not None and longitude is not None:
        mapAx.scatter(longitude, latitude, transform=ccrs.PlateCarree(), color="black", marker="*")
        mapAx.add_feature(cfeat.STATES.with_scale("50m"))
        mapAx.add_feature(plots.USCOUNTIES.with_scale("5m"), edgecolor="gray", linewidth=0.25)
    else:
        mapAx.text(0.5, 0.5, "Location not available", ha="center", va="center", path_effects=[withStroke(linewidth=3, foreground="white")], transform=mapAx.transAxes)
    mapAx.add_feature(cfeat.COASTLINE.with_scale("50m"))
    
    thermodynamicsAx = fig.add_axes([1/20, 1/16, 10/20, 3/16])
    thermodynamicsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotThermoynamics(profileData, thermodynamicsAx)
    thermodynamicsAx.patch.set_alpha(0)

    paramsAx = fig.add_axes([11/20, 1/16, 1/20, 3/16])
    paramsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotParams(profileData, paramsAx)
    paramsAx.patch.set_alpha(0)

    dynamicsAx = fig.add_axes([12/20, 1/16, 7/20, 6/16])
    dynamicsAx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    plotDynamics(profileData, dynamicsAx)

    px = 1/plt.rcParams["figure.dpi"]
    fig.set_size_inches(1920*px, 1080*px)
    width_unit = skew.ax.get_position().width / 10
    height_unit = skew.ax.get_position().height / 10
    tax.set_position([1*width_unit, 14*height_unit, 18*width_unit, 1*height_unit])
    skew.ax.set_position([1*width_unit, 4*height_unit, 10*width_unit, 10*height_unit])
    thermalWindAx.set_position([11*width_unit, 4*height_unit, width_unit, 10*height_unit])
    oldHodoLimits = hodoAx.get_xlim(), hodoAx.get_ylim()
    hodoAx.set_adjustable("datalim")
    hodoAx.set_position([12*width_unit, 9*height_unit, 7*width_unit, 5*height_unit])
    hodoaspect = (hodoAx.get_position().height*1080)/(hodoAx.get_position().width*1920)
    altxmax = ((oldHodoLimits[1][1] - oldHodoLimits[1][0])/hodoaspect)+oldHodoLimits[0][0]
    altymax = ((oldHodoLimits[0][1] - oldHodoLimits[0][0])*hodoaspect)+oldHodoLimits[1][0]
    hodoAx.set_xlim(oldHodoLimits[0][0], np.nanmax([altxmax, oldHodoLimits[0][1]]))
    hodoAx.set_ylim(oldHodoLimits[1][0], np.nanmax([altymax, oldHodoLimits[1][1]]))

    psblHazTypeAx.set_position([12*width_unit, 7*height_unit, 2*width_unit, 2*height_unit])
    partialThicknessAx.set_position([14*width_unit, 7*height_unit, 2*width_unit, 2*height_unit])

    thermodynamicsAx.set_position([1*width_unit, 1*height_unit, 10*width_unit, 3*height_unit])
    paramsAx.set_position([11*width_unit, 1*height_unit, width_unit, 3*height_unit])
    dynamicsAx.set_position([12*width_unit, 1*height_unit, 7*width_unit, 6*height_unit])

    mapAx.set_adjustable("datalim")
    mapAx.set_position([16*width_unit, 7*height_unit, 3*width_unit, 2*height_unit])
    Path(path.dirname(outputPath)).mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath)
    exit()



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
    return data, where, when

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: soundingPlot.py <input> <output>")
        exit()
    if not path.exists(sys.argv[1]):
        print("Input file does not exist!")
        exit()
    soundingData, icao, datetime  = readSharppy(sys.argv[1])
    plotSounding(soundingData, sys.argv[2], icao, datetime)