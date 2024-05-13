from metpy.units import units
from metpy import calc as mpcalc
import numpy as np
import holoviews as hv
from bokeh.models import Range1d, WheelZoomTool, ColumnDataSource, HArea, HoverTool
from bokeh.models.annotations import BoxAnnotation, Label
from functools import reduce
hv.extension('bokeh')

plot_width = 800
plot_height = 800

class skew_t_plot:
    def __init__(self, profileData):
        self.profileData = profileData
        self.skew_t = self.plotSkewT()

    def hook_limit_pan(self, plot, element):
        profileData = self.profileData
        plot.state.select_one(WheelZoomTool).maintain_focus = False
        xlimmin = 10*(np.nanmin(profileData.TEMP.data.to(units.degC).magnitude) // 10)+10
        xlimmax = (10*np.nanmax(profileData.TEMP.data.to(units.degC).magnitude) // 10)+10
        if xlimmax - xlimmin < 110:
            xlimmin = xlimmax - 110
        ylimbot = np.max([np.nanmax(profileData.LEVEL.data.magnitude+5), 1000])
        ylimtop = np.min([np.nanmin(profileData.LEVEL.data.magnitude-5), 100])
        plot.state.x_range = Range1d(xlimmin, xlimmax, bounds=(xlimmin, xlimmax))
        plot.state.y_range = Range1d(ylimbot, ylimtop, bounds='auto')


    def hook_height_label(self, plot, element):
        kmToInterp = np.arange(0, 15001, 1000)
        interpPressures = np.interp(kmToInterp, self.profileData.AGL, self.profileData.LEVEL)
        colorsAndKilometers = {
                (0, 1): "fuchsia",
                (1, 3): "firebrick",
                (3, 6): "limegreen",
                (6, 9): "goldenrod",
                (9, 12): "darkturquoise",
                (12, 15): "darkturquoise"
        }
        for indices, color in reversed(colorsAndKilometers.items()):
            bot, top = indices
            plot.state.add_layout(BoxAnnotation(left=0, right=plot_width/25, bottom=interpPressures[bot], top=interpPressures[top], fill_alpha=0.5, fill_color=color, line_color=color, left_units='screen', right_units='screen'))
            
        for i in [0, 1, 2, 3, 4, 5, 6, 9, 12, 15]:
            if np.nanmax(self.profileData.AGL.data) < i * units.km:
                plot.state.add_layout(Label(x=1, y=np.nanmin(interpPressures), text=f"{np.nanmax(self.profileData.AGL.data).to(units.km).magnitude:.3f} km: {np.nanmin(self.profileData.LEVEL.data).to(units.hPa).magnitude:.1f} hPa",x_units='screen', text_baseline='middle', text_font_size='10px'))
                break
            if i == 0:
                plot.state.add_layout(Label(x=1, y=interpPressures[i], text=f"SFC: {interpPressures[i]:.1f} hPa", x_units='screen', text_baseline='bottom', text_font_size='10px'))
            else:
                plot.state.add_layout(Label(x=1, y=interpPressures[i], text=f"{str(int(i))} km: {interpPressures[i]:.1f} hPa", x_units='screen', text_baseline='bottom', text_font_size='10px'))
        
        
    def hook_inflow_layer(self, plot, element):
        EIL_box = BoxAnnotation(left=plot_width/25, right=3*plot_width/50, bottom=self.profileData.inflowBottom.to(units.hPa).magnitude, top=self.profileData.inflowTop.to(units.hPa).magnitude, fill_alpha=0.2, fill_color='teal', line_color='teal', left_units='screen', right_units='screen')
        plot.state.add_layout(EIL_box)
        eilData = self.profileData.where((self.profileData.LEVEL <= self.profileData.inflowBottom) & (self.profileData.LEVEL >= self.profileData.inflowTop), drop=True)
        eilRH = int(eilData.RH.mean() * 100)
        new_hover = [('Effective Inflow Layer', ''), ('Pressure', f"{self.profileData.inflowBottom.to(units.hPa).magnitude:.1f} - {self.profileData.inflowTop.to(units.hPa).magnitude:.1f} hPa"), ('Height (AGL)', f"{self.profileData.inflowBottom_agl.to(units.meter).magnitude:.1f} - {self.profileData.inflowTop_agl.to(units.meter).magnitude:.1f} m"), ('Relative Humidity', f"{eilRH}%")]
        for hover in plot.state.select(HoverTool):
            if hover.tooltips[2][0] == 'eil_override':
                hover.tooltips = new_hover
                break
        #     skew.ax.text(0.16, profileData.inflowTop.to(units.hPa).magnitude, f"Effective Inflow Layer\n{profileData.inflowBottom.to(units.hPa).magnitude:.1f} - {profileData.inflowTop.to(units.hPa).magnitude:.1f} hPa\nAGL: {int(profileData.inflowBottom_agl.to(units.meter).magnitude)} - {int(profileData.inflowTop_agl.to(units.meter).magnitude)} m\nRH: {eilRH}%", color="teal",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], fontsize=8, clip_on=True, zorder=7, transform=skew.ax.get_yaxis_transform(), alpha=0.7)


    def hook_hover_fix(self, plot, element):
        hovers = plot.state.select(HoverTool)
        for hover in hovers:
            tooltips = []
            for tooltip in hover.tooltips:
                if tooltip[0] == 'Pressure':
                    if 'hPa' not in tooltip[1]:
                        tooltips.append((tooltip[0], f'{tooltip[1]} hPa'))
                elif 'Potential' in tooltip[0]:
                    if 'K' not in tooltip[1]:
                        tooltips.append((tooltip[0], tooltip[1]+'{0.1f} K'))
                elif 'Temperature' in tooltip[0] or tooltip[0] == 'Dew Point':
                    if '°C' not in tooltip[1]:
                        tooltips.append((tooltip[0], tooltip[1]+'{0.1f} °C'))
                elif 'Humidity' in tooltip[0]:
                    if '%' not in tooltip[1]:
                        tooltips.append((tooltip[0], tooltip[1]+'{0.1f} %'))
                elif 'override' in tooltip[0]:
                    tooltips.append((tooltip[0], tooltip[1]))
            if len(tooltips) > 0:
                hover.tooltips = tooltips

    def hook_dgz_area(self, plot, element):
        dgzData = self.profileData.where((self.profileData.LEVEL <= element['DGZ_bottom'][0]*units.hPa), drop=True).where(
            (self.profileData.LEVEL >= element['Pressure'][0]*units.hPa), drop=True)
        

        x1 = (dgzData.DWPT.data.to(units.degC)+dgzData.skewt_offset.data).magnitude
        x2 = (dgzData.TEMP.data.to(units.degC)+dgzData.skewt_offset.data).magnitude
        y = dgzData.LEVEL.data.to(units.hPa).magnitude
        src = ColumnDataSource(data=dict(x1=x1, x2=x2, y=y))
        dgz_shade = HArea(x1='x1', x2='x2', y='y', fill_color='blue', fill_alpha=0.2)
        renderer = plot.state.add_glyph(src, dgz_shade)
        plot.state.add_tools(
            HoverTool(
                renderers=[renderer],
                tooltips=[
                    ('Dendritic Growth Zone', ''),
                    ('Pressure', f"{dgzData.LEVEL[0].data.to(units.hPa).magnitude:.1f} - {dgzData.LEVEL[-1].data.to(units.hPa).magnitude:.1f} hPa"),
                    ('Height (AGL)', f"{dgzData.AGL[0].data.to(units.meter).magnitude:.1f} - {dgzData.AGL[-1].data.to(units.meter).magnitude:.1f} m"),
                    ('Relative Humidity', f"{int(dgzData.RH.data.mean().magnitude*100)}%")
                ]
            )
        )

    def hook_remove_bokeh(self, plot, element):
        plot.state.toolbar.logo = None


    def plotSkewT(self, parcelType="sb"):
        sounding_opts = {
            'ylim': (1000, 100),
            'xlabel' : 'Temperature (°C)',
            'ylabel' : 'Pressure (hPa)',
            'logy':True,
            'width':plot_width,
            'height':plot_height,
            'hooks':[self.hook_limit_pan, self.hook_remove_bokeh, self.hook_hover_fix]
        }
        # Plot data
        skew_t_offset = self.profileData.skewt_offset.data
        temp_curve = hv.Curve((self.profileData.TEMP.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data, self.profileData.TEMP.data.to(units.degC)),
                            kdims=['Skewed_T'], vdims=['Pressure', 'Temperature']).opts(
                                color='red', tools=['hover'], **sounding_opts)
        temp_curve = temp_curve.opts(hooks=[self.hook_height_label])
        # https://docs.bokeh.org/en/latest/docs/reference/models/tools.html#bokeh.models.HoverTool
        dew_curve = hv.Curve((self.profileData.DWPT.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data, self.profileData.DWPT.data.to(units.degC), self.profileData.RH.data.magnitude*100),
                            kdims=['Skewed_T'], vdims=['Pressure', 'Dew Point', 'Relative Humidity']).opts(
                                color='lime', tools=['hover'], **sounding_opts)
        virt_curve = hv.Curve((self.profileData.virtT.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data, self.profileData.virtT.data.to(units.degC)),
                            kdims=['Skewed_T'], vdims=['Pressure', 'Virtual Temperature']).opts(
                                color='red', line_dash='dotted', tools=['hover'], **sounding_opts)
        wetbulb_curve = hv.Curve((self.profileData.wetbulb.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data, self.profileData.wetbulb.data.to(units.degC)),
                            kdims=['Skewed_T'], vdims=['Pressure', 'Wet Bulb Temperature']).opts(
                                color='cyan', line_width=0.5, tools=['hover'], **sounding_opts)
        theta_curve = hv.Curve((self.profileData.potential_temperature.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data,
                                self.profileData.potential_temperature.data.to(units.K)), kdims=['Skewed_T'], vdims=['Pressure', 'Potential Temperature']).opts(
                                    color='mediumslateblue', line_width=0.5, tools=['hover'], **sounding_opts)
        theta_e_curve = hv.Curve((self.profileData.equivalent_potential_temperature.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data,
                                self.profileData.equivalent_potential_temperature.data.to(units.K)), kdims=['Skewed_T'], vdims=['Pressure', 'Equivalent Potential Temperature']).opts(
                                    color='indigo', line_width=0.5, tools=['hover'], **sounding_opts)
        isotherm_min = 10*(np.nanmin(self.profileData.DWPT.data.to(units.degC).magnitude - np.nanmax(skew_t_offset.magnitude)) // 10)+10
        isotherm_max = (10*np.nanmax(self.profileData.TEMP.data.to(units.degC).magnitude) // 10)+30
        isotherms_to_draw = np.arange(isotherm_min, isotherm_max+1, 10)
        isotherms = []
        dry_adiabats = []
        moist_adiabats = []
        for isotherm in isotherms_to_draw:
            isotherm_line = hv.Curve((np.full_like(self.profileData.LEVEL.data, isotherm)*units.degC+skew_t_offset, self.profileData.LEVEL.data), kdims=['Skewed_T'], vdims=['Pressure']).opts(
                color='gray', line_width=0.5, **sounding_opts)
            isotherms.append(isotherm_line)
            this_dry_adiabat = mpcalc.dry_lapse(self.profileData.LEVEL, isotherm*units.degC).to(units.degC)
            dry_daiabat_line = hv.Curve((this_dry_adiabat+skew_t_offset, self.profileData.LEVEL.data), kdims=['Skewed_T'], vdims=['Pressure']).opts(
                color='red', line_width=0.1, **sounding_opts)
            dry_adiabats.append(dry_daiabat_line)
            this_moist_adiabat = mpcalc.moist_lapse(self.profileData.LEVEL, isotherm*units.degC).to(units.degC)
            moist_daiabat_line = hv.Curve((this_moist_adiabat+skew_t_offset, self.profileData.LEVEL.data), kdims=['Skewed_T'], vdims=['Pressure']).opts(
                color='blue', line_width=0.1, **sounding_opts)
            moist_adiabats.append(moist_daiabat_line)
        isotherms = reduce(lambda x, y: x*y, isotherms)
        dry_adiabats = reduce(lambda x, y: x*y, dry_adiabats)
        moist_adiabats = reduce(lambda x, y: x*y, moist_adiabats)

        dgz_isotherms = [hv.Curve((np.full_like(self.profileData.LEVEL.data, isotherm)*units.degC+skew_t_offset, self.profileData.LEVEL.data), kdims=['Skewed_T'], vdims=['Pressure']).opts(
                color='blue', line_dash='dashed', line_width=0.5, **sounding_opts) for isotherm in [-12, -17]]
        dgz_isotherms = reduce(lambda x, y: x*y, dgz_isotherms)

        sfc_dew_label = hv.Text((self.profileData.DWPT.data[0].to(units.degC)+skew_t_offset[0]).magnitude, self.profileData.LEVEL.data[0].magnitude,
                                f"{int((self.profileData.DWPT.data[0]).to(units.degF).magnitude)}°F").opts(color='lime', text_align='right', text_baseline='bottom', text_font_size='12px')
        sfc_wetbulb_label = hv.Text((self.profileData.wetbulb.data[0].to(units.degC)+skew_t_offset[0]).magnitude, self.profileData.LEVEL.data[0].magnitude,
                                f"{int((self.profileData.wetbulb.data[0]).to(units.degF).magnitude)}°F").opts(color='cyan', text_align='center', text_baseline='bottom', text_font_size='12px')
        sfc_temp_label = hv.Text((self.profileData.TEMP.data[0].to(units.degC)+skew_t_offset[0]).magnitude, self.profileData.LEVEL.data[0].magnitude,
                                f"{int((self.profileData.TEMP.data[0]).to(units.degF).magnitude)}°F").opts(color='red', text_align='left', text_baseline='bottom', text_font_size='12px')

        # TODO: add selector for parcel type
        sb_parcel_curve = hv.Curve((self.profileData.sbParcelPath.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data,
                                    self.profileData.sbParcelPath.data.to(units.degC)), kdims=['Skewed_T'], vdims=['Pressure', 'Parcel Temperature']).opts(
                                    color='dimgray', line_dash='dashdot', tools=['hover'], **sounding_opts)


        dcape_curve = hv.Curve((self.profileData.dcape_profile.data.to(units.degC)+skew_t_offset, self.profileData.LEVEL.data,
                                self.profileData.dcape_profile.data.to(units.degC)), kdims=['Skewed_T'], vdims=['Pressure', 'Downdraft Parcel Temperature']).opts(
                                    color='rebeccapurple', line_dash='dashdot', tools=['hover'], **sounding_opts)


        skewt = temp_curve * dew_curve * virt_curve * wetbulb_curve * sb_parcel_curve * isotherms * dry_adiabats * moist_adiabats * dgz_isotherms * sfc_wetbulb_label * sfc_dew_label * sfc_temp_label * theta_curve * theta_e_curve * dcape_curve
        

        # TODO: LFC, LCL, EL labels
        # skew.ax.plot([0, .95], [profileData.attrs[parcelType+"LCL"].magnitude, profileData.attrs[parcelType+"LCL"].magnitude], color="mediumseagreen", linewidth=1, transform=skew.ax.get_yaxis_transform())
        # skew.ax.text(0.875, profileData.attrs[parcelType+"LCL"].magnitude, f"LCL: {profileData.attrs[parcelType+'LCL'].magnitude:.1f} hPa", color="mediumseagreen",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
        # skew.ax.plot([0, .95], [profileData.attrs[parcelType+"LFC"].magnitude, profileData.attrs[parcelType+"LFC"].magnitude], color="darkgoldenrod", linewidth=1, transform=skew.ax.get_yaxis_transform())
        # skew.ax.text(0.875, profileData.attrs[parcelType+"LFC"].magnitude, f"LFC: {profileData.attrs[parcelType+'LFC'].magnitude:.1f} hPa", color="darkgoldenrod",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
        # skew.ax.plot([0, .95], [profileData.attrs[parcelType+"EL"].magnitude, profileData.attrs[parcelType+"EL"].magnitude], color="mediumpurple", linewidth=1, transform=skew.ax.get_yaxis_transform())
        # skew.ax.text(0.875, profileData.attrs[parcelType+"EL"].magnitude, f"EL: {profileData.attrs[parcelType+'EL'].magnitude:.1f} hPa", color="mediumpurple",  ha="left", va="top", path_effects=[withStroke(linewidth=3, foreground="white")], transform=skew.ax.get_yaxis_transform())
    
        dgzsData = (self.profileData.TEMP.data <= -12*units.degC) & (self.profileData.TEMP.data >= -17*units.degC)
        dgzsData = dgzsData.nonzero()[0]
        if len(dgzsData) > 0:
            dgz_bounds = []
            listOfDGZs = []
            dgzBottom = self.profileData.LEVEL.data[dgzsData[0]]
            for i in range(1, len(dgzsData)):
                if dgzsData[i-1] - dgzsData[i] >= 2:
                    dgzTop = self.profileData.LEVEL.data[dgzsData[i-1]]
                    listOfDGZs.append((dgzBottom, dgzTop))
                    dgzBottom = self.profileData.LEVEL.data[dgzsData[i]]
            dgzTop = self.profileData.LEVEL.data[dgzsData[-1]]
            listOfDGZs.append((dgzBottom, dgzTop))
            for dgz in listOfDGZs:
                dgzData = self.profileData.where((self.profileData.LEVEL <= dgz[0]), drop=True).where((self.profileData.LEVEL >= dgz[1]), drop=True)
                upper_line = hv.Curve(([isotherm_min, isotherm_max], [dgzData.LEVEL[-1].data.to(units.hPa).magnitude, dgzData.LEVEL[-1].data.to(units.hPa).magnitude], [dgzData.LEVEL[0].data.to(units.hPa).magnitude, dgzData.LEVEL[0].data.to(units.hPa).magnitude]),
                    kdims=['Skewed_T'], vdims=['Pressure', 'DGZ_bottom']).opts(color='blue', alpha=0.2, tools=['hover'], hooks=[self.hook_dgz_area])
                lower_line = hv.Curve(([isotherm_min, isotherm_max], [dgzData.LEVEL[0].data.to(units.hPa).magnitude, dgzData.LEVEL[0].data.to(units.hPa).magnitude]),
                        kdims=['Skewed_T'], vdims=['Pressure']).opts(color='blue', alpha=0.2)
                dgz_bounds.append(upper_line * lower_line)
            skewt = skewt * reduce(lambda x, y: x*y, dgz_bounds)

        if not np.isnan(self.profileData.inflowBottom):
            EIL_bottom_line = hv.Curve((np.linspace(isotherm_min, isotherm_max, 10, endpoint=True), np.full((10), self.profileData.inflowBottom.to(units.hPa).magnitude), np.zeros((10))),
                kdims=['Skewed_T'], vdims=['Pressure', 'eil_override']).opts(color='teal', alpha=0.2, tools=['hover'], hooks=[self.hook_inflow_layer])
            EIL_top_line = hv.Curve((np.linspace(isotherm_min, isotherm_max, 3, endpoint=True), np.full((3), self.profileData.inflowTop.to(units.hPa).magnitude), np.zeros((10))),
                kdims=['Skewed_T'], vdims=['Pressure', 'eil_override']).opts(color='teal', alpha=0.2, tools=['hover'], hooks=[self.hook_inflow_layer])
            skewt = EIL_bottom_line * EIL_top_line * skewt
        return skewt
