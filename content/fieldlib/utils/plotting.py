# fieldlib/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Ellipse, FancyArrowPatch

def density_plot(field, plane='xy', index=0, kind='amplitude', interactive=False):
    x, y, z = field.volume.get_xyz_axis()

    def _plot(coord_val, plane, kind='amplitude'):
        index = np.abs((x if plane == 'zy' else z) - coord_val).argmin()
        data_vector = field.evaluate_on_volume(plane=plane, index=index)

        # Project field onto polarization vector
        Ex, Ey = data_vector[0], data_vector[1]
        amp = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
        vx = Ex / amp
        vy = Ey / amp

        phi = np.where(vx == 0, np.angle(vy), np.angle(vx))
        
        pol = np.exp(-1j * phi) * np.array([vx,vy])

        if kind == 'amplitude':
            plot_data = amp
            label = 'Amplitude'
        elif kind == 'phase':
            plot_data = phi
            label = 'Phase'
        else:
            raise ValueError("Unsupported kind. Use 'amplitude' or 'phase'.")

        if plane == 'xy':
            X, Y = field.volume.get_xy_grid()
            xaxis, yaxis = 'x', 'y'
        elif plane == 'zy':
            X, Y = field.volume.get_zy_grid()
            xaxis, yaxis = 'z', 'y'
        else:
            raise ValueError("Unsupported plane. Use 'xy' or 'zy'.")

        fig, ax = plt.subplots(figsize=(5, 3))
        if plane == 'xy':
            ax.set_aspect((X.max() - X.min()) / (Y.max() - Y.min()))
        cmap = 'hsv' if kind == 'phase' else 'viridis'
        c = ax.pcolormesh(X, Y, plot_data, shading='auto', cmap=cmap,
                          vmin=-np.pi if kind == 'phase' else None,
                          vmax=np.pi if kind == 'phase' else None)

        if kind == 'phase':
            amplitude = amp
            threshold = 0.2 * np.max(amplitude)
            ax.contour(X, Y, amplitude, levels=[threshold], colors='black', linewidths=1.0)
            ax.contourf(X, Y, amplitude < threshold, levels=[0.5, 1.5], colors='gray', alpha=0.2)

        ax.set_xlabel(f'{xaxis} [a.u.]')
        ax.set_ylabel(f'{yaxis} [a.u.]')
        value = z[index] if plane == 'xy' else x[index]
        ax.set_title(f'{label} in {plane.upper()} Plane ({xaxis}={value:.2e})', pad=10)
        fig.colorbar(c, ax=ax, label=label)
        plt.tight_layout()
        plt.show()

    if interactive:
        slider_z = widgets.FloatSlider(
            value=z[len(z) // 2],
            min=z[0],
            max=z[-1],
            step=(z[-1] - z[0]) / 100,
            readout_format='.2e',
            description='z:'
        )

        slider_x = widgets.FloatSlider(
            value=x[len(x) // 2],
            min=x[0],
            max=x[-1],
            step=(x[-1] - x[0]) / 100,
            readout_format='.2e',
            description='x:'
        )

        button = widgets.Button(description="Calculate")
        reset_button = widgets.Button(description="Reset")
        output_xy = widgets.Output()
        output_zy = widgets.Output()
        output_xy_phase = widgets.Output()
        output_zy_phase = widgets.Output()

        def on_button_clicked(b):
            output_xy.clear_output(wait=True)
            output_zy.clear_output(wait=True)
            output_xy_phase.clear_output(wait=True)
            output_zy_phase.clear_output(wait=True)
            with output_xy:
                _plot(slider_z.value, 'xy', kind='amplitude')
            with output_zy:
                _plot(slider_x.value, 'zy', kind='amplitude')
            with output_xy_phase:
                _plot(slider_z.value, 'xy', kind='phase')
            with output_zy_phase:
                _plot(slider_x.value, 'zy', kind='phase')

        button.on_click(on_button_clicked)

        def on_reset_clicked(b):
            slider_z.value = z[len(z) // 2 - 1]
            slider_x.value = x[len(x) // 2 + 1]

        reset_button.on_click(on_reset_clicked)

        sliders = widgets.HBox([
            widgets.VBox([slider_z], layout=widgets.Layout(align_items='flex-start')),
            widgets.VBox([slider_x], layout=widgets.Layout(align_items='flex-start'))
        ])
        outputs = widgets.VBox([
            widgets.HBox([output_xy, output_zy]),
            widgets.HBox([output_xy_phase, output_zy_phase])
        ])
        display(widgets.VBox([sliders, widgets.HBox([button, reset_button]), outputs]))

        with output_xy:
            _plot(slider_z.value, 'xy', kind='amplitude')
        with output_zy:
            _plot(slider_x.value, 'zy', kind='amplitude')
        with output_xy_phase:
            _plot(slider_z.value, 'xy', kind='phase')
        with output_zy_phase:
            _plot(slider_x.value, 'zy', kind='phase')

    else:
        _plot(z[index], 'xy' if plane != 'zy' else 'zy')

def density_plot_with_polarization(field, plane='xy', index=0, kind='amplitude', interactive=False):
    x, y, z = field.volume.get_xyz_axis()

    def _plot(coord_val, plane, kind='amplitude'):
        index = np.abs((x if plane == 'zy' else z) - coord_val).argmin()  # find the index of the plane closest to the given coordinate value
        data_vector = field.evaluate_on_volume(plane=plane, index=index)

        # Project field onto polarization vector
        Ex, Ey = data_vector[0], data_vector[1]
        amp = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
        vx = Ex / amp
        vy = Ey / amp

        phi = np.where(vx == 0, np.angle(vy), np.angle(vx))
        
        phi_x = np.angle(Ex)
        phi_y = np.angle(Ey)
        delta = phi_y - phi_x

        abs_Ex = np.abs(Ex)
        abs_Ey = np.abs(Ey)
    # Orientation and ellipticity
        numerator = 2 * abs_Ex * abs_Ey * np.cos(delta)
        denominator = abs_Ex**2 - abs_Ey**2
        orientation = 0.5 * np.arctan2(numerator, denominator)
        ellipticity = -2 * abs_Ex * abs_Ey * np.sin(delta) / (abs_Ex**2 + abs_Ey**2 + 1e-20)
       
        if kind == 'amplitude':
            plot_data = amp
            plot_orientations= orientation
            plot_ellipses = ellipticity
            label = 'Amplitude'
        elif kind == 'phase':
            plot_data = phi
            label = 'Phase'
        elif kind == 'orientation':
            plot_data = orientation
            label = 'Orientation'
        elif kind == 'ellipticity':
            plot_data = ellipticity
            label = 'Ellipticity'
        else:
            raise ValueError("Unsupported kind. Use 'amplitude', 'phase', 'orientation' or 'ellipticity'.")

        if plane == 'xy':
            X, Y = field.volume.get_xy_grid()
            XYratio= (X.max() - X.min()) / (Y.max() - Y.min())
            xaxis, yaxis = 'x', 'y'
        elif plane == 'zy':
            X, Y = field.volume.get_zy_grid()
            XYratio= (X.max() - X.min()) / (Y.max() - Y.min())
            xaxis, yaxis = 'z', 'y'
        else:
            raise ValueError("Unsupported plane. Use 'xy' or 'zy'.")
        
        N = 11
        # For each axis, pick indices that split the range evenly
        ny, nx = X.shape
        row_idx = np.linspace(0, ny - 1, N, dtype=int)
        col_idx = np.linspace(0, nx - 1, N, dtype=int)
        X_coarse = X[np.ix_(row_idx, col_idx)]
        Y_coarse = Y[np.ix_(row_idx, col_idx)]

        orientation_coarse=orientation[np.ix_(row_idx, col_idx)]
        ellipticity_coarse=ellipticity[np.ix_(row_idx, col_idx)]

        fig, ax= plt.subplots(figsize=(5, 3))

        if plane == 'xy':
            ax.set_aspect((X.max() - X.min()) / (Y.max() - Y.min()))
        cmap = 'hsv' if kind == 'phase' or kind == 'orientation' else 'viridis'
        if kind == 'phase' or kind == 'orientation':
            vmin = -np.pi
            vmax = np.pi
        elif kind == 'ellipticity':
            vmin = -1
            vmax = 1
        else:
            vmin = None
            vmax = None
        c = ax.pcolormesh(X, Y, plot_data, shading='auto', cmap=cmap,
                          vmin=vmin,
                          vmax=vmax,
                          alpha=0.65)
        
        ax.set_xlim(X.min(),X.max())
        ax.set_ylim(Y.min(),Y.max())
        
        if (kind == 'amplitude' or kind == 'phase') and plane == 'xy':
            dx =  X_coarse[0, 1] -  X_coarse[0, 0]
            dy =  Y_coarse[1, 0] -  Y_coarse[0, 0]
            #dx=dx/XYratio

            ellipse_width = 0.45 * dx
            ellipse_height = 0.45 * dy

            for i in range(N):
                for j in range(N):    
                    cx, cy = X_coarse[i, j], Y_coarse[i, j]
                    angle = orientation_coarse[i, j]
                    major = ellipse_width
                    minor = ellipse_height * abs(ellipticity_coarse[i, j])
                    if np.abs( ellipticity_coarse[i, j])<0.05:
                        edgecolor='white'
                    elif np.sign(ellipticity_coarse[i, j])==1:
                        edgecolor='black'
                    else:
                        edgecolor='white'

                    ell = Ellipse((cx, cy),
                                width=major,
                                height=minor,
                                angle=angle/np.pi*180,
                                edgecolor=edgecolor,
                                facecolor='none',
                                lw=0.5)
                    ax.add_patch(ell)  

            #         # arrow
            #         direction = np.sign(ellipticity_coarse[i, j])
            #         theta = orientation_coarse[i, j]
            #         arrow_length = min(dx, dy) * 0.2
            #         if direction > 0:
            #             # CW: angle + 90 deg
            #             arrow_theta = theta + np.pi / 2
            #         else:
            #             # CCW: angle - 90 deg
            #             arrow_theta = theta - np.pi / 2
            # # Start point at edge of ellipse along major axis
            #         ex = cx -(minor / 2) * np.sin(theta)
            #         ey = cy + (minor / 2) * np.cos(theta)

            #         # Arrow direction
            #         dx_arrow = arrow_length * np.sin(arrow_theta)
            #         dy_arrow = -arrow_length * np.cos(arrow_theta)
                

            #         arrow = FancyArrowPatch(
            #             (ex, ey),
            #             (ex + dx_arrow, ey + dy_arrow),
            #             arrowstyle='->',
            #             mutation_scale=7,
            #             color='white',
            #             lw=1
            #         )
            #         ax.add_patch(arrow)
            

        if kind == 'phase' or kind == 'orientation' or kind == 'ellipticity':
            amplitude = amp
            threshold = 0.2 * np.max(amplitude)
            ax.contour(X, Y, amplitude, levels=[threshold], colors='black', linewidths=1.0)
            ax.contourf(X, Y, amplitude < threshold, levels=[0.5, 1.5], colors='gray', alpha=0.2)

        ax.set_xlabel(f'{xaxis} [m]')
        ax.set_ylabel(f'{yaxis} [m]')
        value = z[index] if plane == 'xy' else x[index]
        ax.set_title(f'{label} in {plane.upper()} Plane ({xaxis}={value:.2e})', pad=10)
        fig.colorbar(c, ax=ax, label=label)

        plt.tight_layout()
        plt.show()

    if interactive:
        slider_z = widgets.FloatSlider(
            value=z[len(z) // 2],
            min=z[0],
            max=z[-1],
            step=(z[-1] - z[0]) / 100,
            readout_format='.2e',
            description='z:'
        )

        slider_x = widgets.FloatSlider(
            value=x[len(x) // 2],
            min=x[0],
            max=x[-1],
            step=(x[-1] - x[0]) / 100,
            readout_format='.2e',
            description='x:'
        )

        button = widgets.Button(description="Calculate")
        reset_button = widgets.Button(description="Reset")
        output_xy = widgets.Output()
        output_zy = widgets.Output()
        output_xy_phase = widgets.Output()
        output_zy_phase = widgets.Output()
        # output_xy_orientation = widgets.Output()
        # output_zy_orientation = widgets.Output()
        output_xy_ellipticity = widgets.Output()
        output_zy_ellipticity = widgets.Output()
        def on_button_clicked(b):
            output_xy.clear_output(wait=True)
            output_zy.clear_output(wait=True)
            output_xy_phase.clear_output(wait=True)
            output_zy_phase.clear_output(wait=True)
            # output_xy_orientation.clear_output(wait=True)
            # output_zy_orientation.clear_output(wait=True)
            output_xy_ellipticity.clear_output(wait=True)
            output_zy_ellipticity.clear_output(wait=True)

            with output_xy:
                _plot(slider_z.value, 'xy', kind='amplitude')
            with output_zy:
                _plot(slider_x.value, 'zy', kind='amplitude')
            with output_xy_phase:
                _plot(slider_z.value, 'xy', kind='phase')
            with output_zy_phase:
                _plot(slider_x.value, 'zy', kind='phase')
            # with output_xy_orientation:
            #     _plot(slider_z.value, 'xy', kind='orientation')
            # with output_zy_orientation:
            #     _plot(slider_x.value, 'zy', kind='orientation')
            with output_xy_ellipticity:
                _plot(slider_z.value, 'xy', kind='ellipticity')
            with output_zy_ellipticity:
                _plot(slider_x.value, 'zy', kind='ellipticity')

        button.on_click(on_button_clicked)

        def on_reset_clicked(b):
            slider_z.value = z[len(z) // 2 - 1]
            slider_x.value = x[len(x) // 2 + 1]

        reset_button.on_click(on_reset_clicked)

        sliders = widgets.HBox([
            widgets.VBox([slider_z], layout=widgets.Layout(align_items='flex-start')),
            widgets.VBox([slider_x], layout=widgets.Layout(align_items='flex-start'))
        ])
        outputs = widgets.VBox([
            widgets.HBox([output_xy, output_zy]),
            widgets.HBox([output_xy_phase, output_zy_phase]),
            # widgets.HBox([output_xy_orientation, output_zy_orientation]),
            widgets.HBox([output_xy_ellipticity, output_zy_ellipticity])
        ])
        display(widgets.VBox([sliders, widgets.HBox([button, reset_button]), outputs]))

        with output_xy:
            _plot(slider_z.value, 'xy', kind='amplitude')
        with output_zy:
            _plot(slider_x.value, 'zy', kind='amplitude')
        with output_xy_phase:
            _plot(slider_z.value, 'xy', kind='phase')
        with output_zy_phase:
            _plot(slider_x.value, 'zy', kind='phase')
        # with output_xy_orientation:
        #     _plot(slider_z.value, 'xy', kind='orientation')
        # with output_zy_orientation:
        #     _plot(slider_x.value, 'zy', kind='orientation')
        with output_xy_ellipticity:
            _plot(slider_z.value, 'xy', kind='ellipticity')
        with output_zy_ellipticity:
            _plot(slider_x.value, 'zy', kind='ellipticity')
    else:
        _plot(z[index], 'xy' if plane != 'zy' else 'zy')


# def density_plot_with_ellipses(field, plane='xy', index=0, kind='amplitude', interactive=False):
#     x, y, z = field.volume.get_xyz_axis()

#     def _plot(coord_val, plane, kind='amplitude', show_ellipses=True):
#         index = np.abs((x if plane == 'zy' else z) - coord_val).argmin()
#         data_vector = field.evaluate_on_volume(plane=plane, index=index)

#         # Project field onto polarization vector
#         Ex, Ey = data_vector[0], data_vector[1]
#         amp = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2)
#         vx = Ex / amp
#         vy = Ey / amp

#         phi = np.where(vx == 0, np.angle(vy), np.angle(vx))
        
#         phi_x = np.angle(Ex)
#         phi_y = np.angle(Ey)
#         delta = phi_y - phi_x

#         abs_Ex = np.abs(Ex)
#         abs_Ey = np.abs(Ey)
#     # Orientation and ellipticity
#         numerator = 2 * abs_Ex * abs_Ey * np.cos(delta)
#         denominator = abs_Ex**2 - abs_Ey**2
#         orientation = 0.5 * np.arctan2(numerator, denominator)
#         ellipticity = -2 * abs_Ex * abs_Ey * np.sin(delta) / (abs_Ex**2 + abs_Ey**2 + 1e-20)
        
#         if kind == 'amplitude':
#             plot_data = amp
#             label = 'Amplitude'
#         elif kind == 'phase':
#             plot_data = phi
#             label = 'Phase'
#         elif kind == 'orientation':
#             plot_data = orientation
#             label = 'Orientation'
#         elif kind == 'ellipticity':
#             plot_data = ellipticity
#             label = 'Ellipticity'
#         else:
#             raise ValueError("Unsupported kind. Use 'amplitude', 'phase', 'orientation' or 'ellipticity'.")

#         if plane == 'xy':
#             X, Y = field.volume.get_xy_grid()
#             xaxis, yaxis = 'x', 'y'
#         elif plane == 'zy':
#             X, Y = field.volume.get_zy_grid()
#             xaxis, yaxis = 'z', 'y'
#         else:
#             raise ValueError("Unsupported plane. Use 'xy' or 'zy'.")

#         fig, ax = plt.subplots(figsize=(5, 3))
#         if plane == 'xy':
#             ax.set_aspect((X.max() - X.min()) / (Y.max() - Y.min()))
#         cmap = 'hsv' if kind == 'phase' or kind == 'orientation' else 'viridis'
#         if kind == 'phase' or kind == 'orientation':
#             vmin = -np.pi
#             vmax = np.pi
#         elif kind == 'ellipticity':
#             vmin = -1
#             vmax = 1
#         else:
#             vmin = None
#             vmax = None
#         c = ax.pcolormesh(X, Y, plot_data, shading='auto', cmap=cmap,
#                           vmin=vmin,
#                           vmax=vmax)

#         if kind == 'phase' or kind == 'orientation' or kind == 'ellipticity':
#             amplitude = amp
#             threshold = 0.2 * np.max(amplitude)
#             ax.contour(X, Y, amplitude, levels=[threshold], colors='black', linewidths=1.0)
#             ax.contourf(X, Y, amplitude < threshold, levels=[0.5, 1.5], colors='gray', alpha=0.2)

#         ax.set_xlabel(f'{xaxis} [m]')
#         ax.set_ylabel(f'{yaxis} [m]')
#         value = z[index] if plane == 'xy' else x[index]
#         ax.set_title(f'{label} in {plane.upper()} Plane ({xaxis}={value:.2e})', pad=10)
#         fig.colorbar(c, ax=ax, label=label)
#         plt.tight_layout()
#         plt.show()

#     if interactive:
#         slider_z = widgets.FloatSlider(
#             value=z[len(z) // 2],
#             min=z[0],
#             max=z[-1],
#             step=(z[-1] - z[0]) / 100,
#             readout_format='.2e',
#             description='z:'
#         )

#         slider_x = widgets.FloatSlider(
#             value=x[len(x) // 2],
#             min=x[0],
#             max=x[-1],
#             step=(x[-1] - x[0]) / 100,
#             readout_format='.2e',
#             description='x:'
#         )

#         button = widgets.Button(description="Calculate")
#         reset_button = widgets.Button(description="Reset")
#         output_xy = widgets.Output()
#         output_zy = widgets.Output()
#         output_xy_phase = widgets.Output()
#         output_zy_phase = widgets.Output()
#         output_xy_orientation = widgets.Output()
#         output_zy_orientation = widgets.Output()
#         output_xy_ellipticity = widgets.Output()
#         output_zy_ellipticity = widgets.Output()
#         def on_button_clicked(b):
#             output_xy.clear_output(wait=True)
#             output_zy.clear_output(wait=True)
#             output_xy_phase.clear_output(wait=True)
#             output_zy_phase.clear_output(wait=True)
#             output_xy_orientation.clear_output(wait=True)
#             output_zy_orientation.clear_output(wait=True)
#             output_xy_ellipticity.clear_output(wait=True)
#             output_zy_ellipticity.clear_output(wait=True)

#             with output_xy:
#                 _plot(slider_z.value, 'xy', kind='amplitude')
#             with output_zy:
#                 _plot(slider_x.value, 'zy', kind='amplitude')
#             with output_xy_phase:
#                 _plot(slider_z.value, 'xy', kind='phase')
#             with output_zy_phase:
#                 _plot(slider_x.value, 'zy', kind='phase')
#             with output_xy_orientation:
#                 _plot(slider_z.value, 'xy', kind='orientation')
#             with output_zy_orientation:
#                 _plot(slider_x.value, 'zy', kind='orientation')
#             with output_xy_ellipticity:
#                 _plot(slider_z.value, 'xy', kind='ellipticity')
#             with output_zy_ellipticity:
#                 _plot(slider_x.value, 'zy', kind='ellipticity')

#         button.on_click(on_button_clicked)

#         def on_reset_clicked(b):
#             slider_z.value = z[len(z) // 2 - 1]
#             slider_x.value = x[len(x) // 2 + 1]

#         reset_button.on_click(on_reset_clicked)

#         sliders = widgets.HBox([
#             widgets.VBox([slider_z], layout=widgets.Layout(align_items='flex-start')),
#             widgets.VBox([slider_x], layout=widgets.Layout(align_items='flex-start'))
#         ])
#         outputs = widgets.VBox([
#             widgets.HBox([output_xy, output_zy]),
#             widgets.HBox([output_xy_phase, output_zy_phase]),
#             widgets.HBox([output_xy_orientation, output_zy_orientation]),
#             widgets.HBox([output_xy_ellipticity, output_zy_ellipticity])
#         ])
#         display(widgets.VBox([sliders, widgets.HBox([button, reset_button]), outputs]))

#         with output_xy:
#             _plot(slider_z.value, 'xy', kind='amplitude')
#         with output_zy:
#             _plot(slider_x.value, 'zy', kind='amplitude')
#         with output_xy_phase:
#             _plot(slider_z.value, 'xy', kind='phase')
#         with output_zy_phase:
#             _plot(slider_x.value, 'zy', kind='phase')
#         with output_xy_orientation:
#             _plot(slider_z.value, 'xy', kind='orientation')
#         with output_zy_orientation:
#             _plot(slider_x.value, 'zy', kind='orientation')
#         with output_xy_ellipticity:
#             _plot(slider_z.value, 'xy', kind='ellipticity')
#         with output_zy_ellipticity:
#             _plot(slider_x.value, 'zy', kind='ellipticity')
#     else:
#         _plot(z[index], 'xy' if plane != 'zy' else 'zy')

