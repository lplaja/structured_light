# fieldlib/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

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
        index = np.abs((x if plane == 'zy' else z) - coord_val).argmin()
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
            xaxis, yaxis = 'x', 'y'
        elif plane == 'zy':
            X, Y = field.volume.get_zy_grid()
            xaxis, yaxis = 'z', 'y'
        else:
            raise ValueError("Unsupported plane. Use 'xy' or 'zy'.")

        fig, ax = plt.subplots(figsize=(5, 3))
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
                          vmax=vmax)

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
        output_xy_orientation = widgets.Output()
        output_zy_orientation = widgets.Output()
        output_xy_ellipticity = widgets.Output()
        output_zy_ellipticity = widgets.Output()
        def on_button_clicked(b):
            output_xy.clear_output(wait=True)
            output_zy.clear_output(wait=True)
            output_xy_phase.clear_output(wait=True)
            output_zy_phase.clear_output(wait=True)
            output_xy_orientation.clear_output(wait=True)
            output_zy_orientation.clear_output(wait=True)
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
            with output_xy_orientation:
                _plot(slider_z.value, 'xy', kind='orientation')
            with output_zy_orientation:
                _plot(slider_x.value, 'zy', kind='orientation')
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
            widgets.HBox([output_xy_orientation, output_zy_orientation]),
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
        with output_xy_orientation:
            _plot(slider_z.value, 'xy', kind='orientation')
        with output_zy_orientation:
            _plot(slider_x.value, 'zy', kind='orientation')
        with output_xy_ellipticity:
            _plot(slider_z.value, 'xy', kind='ellipticity')
        with output_zy_ellipticity:
            _plot(slider_x.value, 'zy', kind='ellipticity')
    else:
        _plot(z[index], 'xy' if plane != 'zy' else 'zy')


