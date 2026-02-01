import sys

import nidaqmx
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget, QCheckBox,
    QFileDialog
)
from nidaqmx.system import System
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression


class DAQThread(QThread):
    data_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self, channels, sampling_rate, samples_per_channel):
        super().__init__()
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.samples_per_channel = samples_per_channel
        self.running = False

    def run(self):
        self.running = True
        try:
            with nidaqmx.Task() as task:
                for channel in self.channels:
                    task.ai_channels.add_ai_voltage_chan(channel)
                task.timing.cfg_samp_clk_timing(self.sampling_rate,
                                                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

                while self.running:
                    try:
                        data = task.read(number_of_samples_per_channel=self.samples_per_channel)
                        if isinstance(data[0], list):
                            data_dict = {chan: np.array(data[i]) for i, chan in enumerate(self.channels)}
                        else:
                            data_dict = {self.channels[0]: np.array(data)}
                        time_array = np.linspace(0, len(data_dict[self.channels[0]]) / self.sampling_rate,
                                                 len(data_dict[self.channels[0]]))
                        self.data_ready.emit(time_array, data_dict)
                    except nidaqmx.DaqError as e:
                        if e.error_code == -200279:
                            print(
                                f"DAQ Error -200279: The application is not able to keep up with the hardware acquisition. Increasing the buffer size or reading data more frequently might help.")
                            continue
                        else:
                            print(f"Unexpected DAQ Error: {e}")
                            break
        except nidaqmx.DaqError as e:
            self.data_ready.emit(np.array([]), {})  # Emit empty arrays to indicate an error
            print(f"Failed to initialize DAQ task: {e}")

    def stop(self):
        self.running = False


class SerialCommTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Renishaw DAQ Acquisition")
        self.setGeometry(100, 100, 1400, 1200)  # Increased height for new plots

        # Initialize instance variables
        self.daq_thread_tab3 = None
        self.sampling_rate = 1000
        self.samples_on_plot = 1000
        self.buffer_size = 1000
        self.daq_data_buffers = {1: np.array([]), 2: np.array([])}
        self.time_data_buffer = np.array([])
        self.processed_data_ch1 = np.array([])
        self.processed_data_ch2 = np.array([])
        self.distance_data = np.array([])
        self.residual_data = np.array([])  # New buffer for residuals
        self.fft_freq = np.array([])  # New buffer for FFT frequencies
        self.fft_magnitude = np.array([])  # New buffer for FFT magnitudes
        self.daq_running = False  # Track DAQ state
        self.last_unwrapped_value = None  # For phase unwrapping continuity
        self.phase_offset = 0.0  # Add phase offset for zeroing
        self.downsample_points = 10000  # Default downsample points
        self.save_time_length = 10.0  # Default save time length in seconds
        self.save_start_time = None  # Track when saving started
        self.saved_distance_data = np.array([])  # Buffer for saving data
        self.saved_time_data = np.array([])  # Buffer for saving time data
        self.is_saving = False  # Track if currently saving

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Add DAQ tab
        daq_tab = QWidget()
        self.tab_widget.addTab(daq_tab, "DAQ Acquisition")
        daq_layout = self.create_dfb_driver_layout()
        daq_tab.setLayout(daq_layout)

    # def process_channel_data(self, data):
    #     """Process channel data based on voltage range"""
    #     if len(data) == 0:
    #         return data
    #
    #     gap = (np.max(data) - np.min(data)) / 2
    #
    #     offset = (np.max(data) + np.min(data)) / 2
    #
    #     if gap > 0.1:
    #         processed_data = (data - offset) / gap
    #     else:
    #         # Subtract 2.51 from all data points
    #         processed_data = data - 2.51
    #
    #     return processed_data

    def process_channel_data(self, data):
        """Process channel data based on voltage range"""
        if len(data) == 0:
            return data

        gap = (np.max(data) - np.min(data)) / 2

        offset = (np.max(data) + np.min(data)) / 2

        processed_data = (data - offset) / gap

        return processed_data

    def incremental_unwrap_array(self, raw_phase_array: np.ndarray) -> np.ndarray:
        """
        Incrementally unwrap the phase array in a single pass.
        """
        n = len(raw_phase_array)
        if n == 0:
            return np.array([])

        unwrapped_array = np.zeros_like(raw_phase_array)

        # Handle the first sample of the new chunk
        if self.last_unwrapped_value is None:
            # No previous chunk
            unwrapped_array[0] = raw_phase_array[0]
        else:
            # Offset so that new chunk's first sample lines up with last_unwrapped_value
            offset = self.last_unwrapped_value - raw_phase_array[0]
            unwrapped_array[0] = raw_phase_array[0] + offset

        # Single-pass unwrapping for the rest of the chunk
        for i in range(1, n):
            diff = raw_phase_array[i] - raw_phase_array[i - 1]

            if diff > np.pi:
                diff -= 2.0 * np.pi
            elif diff < -np.pi:
                diff += 2.0 * np.pi

            unwrapped_array[i] = unwrapped_array[i - 1] + diff

        # Update the global continuity reference
        self.last_unwrapped_value = unwrapped_array[-1]

        return unwrapped_array

    def calculate_residuals_and_fft(self, distance_data, time_array):
        """Calculate residuals from linear fit and compute FFT spectrum"""
        if len(distance_data) < 10:  # Need minimum points for fitting
            return np.array([]), np.array([]), np.array([])

        # Perform linear regression
        X = time_array.reshape(-1, 1)
        y = distance_data

        reg = LinearRegression().fit(X, y)
        linear_fit = reg.predict(X)

        # Calculate residuals
        residuals = distance_data - linear_fit

        # Calculate FFT of residuals
        if len(residuals) > 1:
            fft_result = fft(residuals)
            fft_freq = fftfreq(len(residuals), 1 / self.sampling_rate)

            # Take only positive frequencies and magnitude
            positive_freq_mask = fft_freq > 0
            fft_freq_positive = fft_freq[positive_freq_mask]
            fft_magnitude = np.abs(fft_result[positive_freq_mask])

            return residuals, fft_freq_positive, fft_magnitude

        return residuals, np.array([]), np.array([])

    def create_dfb_driver_layout(self):
        main_layout = QVBoxLayout()

        # NI-DAQ ai Group Box
        daq_ai_groupbox = QGroupBox("NI-DAQ ai")
        daq_ai_layout = QVBoxLayout()

        # Channel selection layout
        channel_layout_tab3 = QHBoxLayout()
        channel_layout_tab3.addWidget(QLabel('Select Analog Channels:'))
        self.channel_select_dropdown_tab3_1 = QComboBox()
        self.channel_select_dropdown_tab3_2 = QComboBox()
        self.channel_select_dropdown_tab3_1.addItems(self.get_available_channels())
        self.channel_select_dropdown_tab3_2.addItems(self.get_available_channels())
        self.channel_select_dropdown_tab3_1.setFixedWidth(200)
        self.channel_select_dropdown_tab3_2.setFixedWidth(200)
        channel_layout_tab3.addWidget(self.channel_select_dropdown_tab3_1)
        channel_layout_tab3.addWidget(self.channel_select_dropdown_tab3_2)

        # Settings layout
        settings_layout_tab3 = QHBoxLayout()
        sampling_rate_layout_tab3 = QVBoxLayout()
        sampling_rate_layout_tab3.addWidget(QLabel('Sampling Rate:'))
        self.sampling_rate_input_tab3 = QLineEdit()
        self.sampling_rate_input_tab3.setFixedWidth(200)
        sampling_rate_layout_tab3.addWidget(self.sampling_rate_input_tab3)
        settings_layout_tab3.addLayout(sampling_rate_layout_tab3)

        samples_per_channel_layout_tab3 = QVBoxLayout()
        samples_per_channel_layout_tab3.addWidget(QLabel('Samples per Channel:'))
        self.samples_per_channel_input_tab3 = QLineEdit()
        self.samples_per_channel_input_tab3.setFixedWidth(200)
        samples_per_channel_layout_tab3.addWidget(self.samples_per_channel_input_tab3)
        settings_layout_tab3.addLayout(samples_per_channel_layout_tab3)

        samples_on_plot_layout_tab3 = QVBoxLayout()
        samples_on_plot_layout_tab3.addWidget(QLabel('Samples on Plot:'))
        self.samples_on_plot_input_tab3 = QLineEdit()
        self.samples_on_plot_input_tab3.setFixedWidth(200)
        samples_on_plot_layout_tab3.addWidget(self.samples_on_plot_input_tab3)
        settings_layout_tab3.addLayout(samples_on_plot_layout_tab3)

        save_time_layout_tab3 = QVBoxLayout()
        save_time_layout_tab3.addWidget(QLabel('Save Time Length (s):'))
        self.save_time_input_tab3 = QLineEdit()
        self.save_time_input_tab3.setText('10.0')  # Default value
        self.save_time_input_tab3.setFixedWidth(200)
        save_time_layout_tab3.addWidget(self.save_time_input_tab3)
        settings_layout_tab3.addLayout(save_time_layout_tab3)

        downsample_layout_tab3 = QVBoxLayout()
        downsample_layout_tab3.addWidget(QLabel('Downsample Points:'))
        self.downsample_input_tab3 = QLineEdit()
        self.downsample_input_tab3.setText('10000')  # Default value
        self.downsample_input_tab3.setFixedWidth(200)
        downsample_layout_tab3.addWidget(self.downsample_input_tab3)
        settings_layout_tab3.addLayout(downsample_layout_tab3)

        daq_ai_layout.addLayout(channel_layout_tab3)
        daq_ai_layout.addLayout(settings_layout_tab3)

        # Plot visibility checkboxes
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(QLabel('Show Plots:'))

        self.show_ch1_checkbox = QCheckBox('Channel 1')
        self.show_ch1_checkbox.setChecked(True)
        self.show_ch1_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        checkbox_layout.addWidget(self.show_ch1_checkbox)

        self.show_ch2_checkbox = QCheckBox('Channel 2')
        self.show_ch2_checkbox.setChecked(True)
        self.show_ch2_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        checkbox_layout.addWidget(self.show_ch2_checkbox)

        self.show_xy_checkbox = QCheckBox('XY Plot')
        self.show_xy_checkbox.setChecked(True)
        self.show_xy_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        checkbox_layout.addWidget(self.show_xy_checkbox)

        self.show_distance_checkbox = QCheckBox('Distance')
        self.show_distance_checkbox.setChecked(True)
        self.show_distance_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        checkbox_layout.addWidget(self.show_distance_checkbox)

        self.show_residual_checkbox = QCheckBox('Residual')
        self.show_residual_checkbox.setChecked(True)
        self.show_residual_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        checkbox_layout.addWidget(self.show_residual_checkbox)

        self.show_fft_checkbox = QCheckBox('FFT')
        self.show_fft_checkbox.setChecked(True)
        self.show_fft_checkbox.stateChanged.connect(self.toggle_plot_visibility)
        checkbox_layout.addWidget(self.show_fft_checkbox)

        daq_ai_layout.addLayout(checkbox_layout)

        # Create horizontal layout for the two time-domain plots (First row)
        time_plots_layout = QHBoxLayout()

        # Channel 1 time-domain plot
        self.daq_plot_widget_tab3_1 = pg.PlotWidget()
        self.daq_plot_widget_tab3_1.setBackground('w')  # White background
        self.daq_plot_widget_tab3_1.setLabel('left', 'Voltage', units='V', color='black')
        self.daq_plot_widget_tab3_1.setLabel('bottom', 'Time', units='s', color='black')
        self.daq_plot_widget_tab3_1.setTitle('Channel 1', color='black')
        self.daq_plot_widget_tab3_1.getAxis('left').setStyle(tickFont=QFont("Arial", 22))
        self.daq_plot_widget_tab3_1.getAxis('bottom').setStyle(tickFont=QFont("Arial", 22))
        time_plots_layout.addWidget(self.daq_plot_widget_tab3_1)  # Add Channel 1 to layout

        # Channel 2 time-domain plot
        self.daq_plot_widget_tab3_2 = pg.PlotWidget()
        self.daq_plot_widget_tab3_2.setBackground('w')  # White background
        self.daq_plot_widget_tab3_2.setLabel('left', 'Voltage', units='V', color='black')
        self.daq_plot_widget_tab3_2.setLabel('bottom', 'Time', units='s', color='black')
        self.daq_plot_widget_tab3_2.setTitle('Channel 2', color='black')
        self.daq_plot_widget_tab3_2.getAxis('left').setStyle(tickFont=QFont("Arial", 22))
        self.daq_plot_widget_tab3_2.getAxis('bottom').setStyle(tickFont=QFont("Arial", 22))
        time_plots_layout.addWidget(self.daq_plot_widget_tab3_2)  # Ensure Channel 2 is added to layout

        daq_ai_layout.addLayout(time_plots_layout)

        # Create horizontal layout for XY plot and distance plot (Second row)
        analysis_plots_layout = QHBoxLayout()

        # XY plot
        self.xy_plot_widget = pg.PlotWidget()
        self.xy_plot_widget.setBackground('w')  # White background
        self.xy_plot_widget.setLabel('left', 'Channel 2 Processed', units='V', color='black')
        self.xy_plot_widget.setLabel('bottom', 'Channel 1 Processed', units='V', color='black')
        self.xy_plot_widget.setTitle('Processed Channel 1 vs Channel 2', color='black')
        self.xy_plot_widget.getAxis('left').setStyle(tickFont=QFont("Arial", 22))
        self.xy_plot_widget.getAxis('bottom').setStyle(tickFont=QFont("Arial", 22))
        analysis_plots_layout.addWidget(self.xy_plot_widget)

        # Distance plot
        self.distance_plot_widget = pg.PlotWidget()
        self.distance_plot_widget.setBackground('w')  # White background
        self.distance_plot_widget.setLabel('left', 'Distance', units='nm', color='black')
        self.distance_plot_widget.setLabel('bottom', 'Time', units='s', color='black')
        self.distance_plot_widget.setTitle('Calculated Distance', color='black')
        self.distance_plot_widget.getAxis('left').setStyle(tickFont=QFont("Arial", 22))
        self.distance_plot_widget.getAxis('bottom').setStyle(tickFont=QFont("Arial", 22))
        analysis_plots_layout.addWidget(self.distance_plot_widget)

        daq_ai_layout.addLayout(analysis_plots_layout)

        # Create horizontal layout for residual and FFT plots (Third row)
        residual_plots_layout = QHBoxLayout()

        # Residual plot
        self.residual_plot_widget = pg.PlotWidget()
        self.residual_plot_widget.setBackground('w')  # White background
        self.residual_plot_widget.setLabel('left', 'Residual', units='nm', color='black')
        self.residual_plot_widget.setLabel('bottom', 'Time', units='s', color='black')
        self.residual_plot_widget.setTitle('Distance Residuals (Distance - Linear Fit)', color='black')
        self.residual_plot_widget.getAxis('left').setStyle(tickFont=QFont("Arial", 22))
        self.residual_plot_widget.getAxis('bottom').setStyle(tickFont=QFont("Arial", 22))
        residual_plots_layout.addWidget(self.residual_plot_widget)

        # FFT plot
        self.fft_plot_widget = pg.PlotWidget()
        self.fft_plot_widget.setBackground('w')  # White background
        self.fft_plot_widget.setLabel('left', 'Magnitude', color='black')
        self.fft_plot_widget.setLabel('bottom', 'Frequency', units='Hz', color='black')
        self.fft_plot_widget.setTitle('FFT Spectrum of Residuals', color='black')
        self.fft_plot_widget.getAxis('left').setStyle(tickFont=QFont("Arial", 22))
        self.fft_plot_widget.getAxis('bottom').setStyle(tickFont=QFont("Arial", 22))
        residual_plots_layout.addWidget(self.fft_plot_widget)

        daq_ai_layout.addLayout(residual_plots_layout)

        # Button layout
        button_layout_tab3 = QHBoxLayout()

        # Single toggle button for start/stop DAQ
        self.toggle_daq_button_tab3 = QPushButton('Start DAQ')
        self.toggle_daq_button_tab3.clicked.connect(self.toggle_daq_tab3)
        button_layout_tab3.addWidget(self.toggle_daq_button_tab3)

        self.clear_plot_button_tab3 = QPushButton('Clear Plot')
        self.clear_plot_button_tab3.clicked.connect(self.clear_plot_tab3)
        button_layout_tab3.addWidget(self.clear_plot_button_tab3)

        # Add Zero Distance button
        self.zero_distance_button_tab3 = QPushButton('Zero Distance')
        self.zero_distance_button_tab3.clicked.connect(self.zero_distance)
        button_layout_tab3.addWidget(self.zero_distance_button_tab3)

        # Add Save Distance Data button
        self.save_data_button_tab3 = QPushButton('Save Distance Data')
        self.save_data_button_tab3.clicked.connect(self.save_distance_data)
        button_layout_tab3.addWidget(self.save_data_button_tab3)

        daq_ai_layout.addLayout(button_layout_tab3)
        daq_ai_groupbox.setLayout(daq_ai_layout)

        main_layout.addWidget(daq_ai_groupbox)
        return main_layout

    def zero_distance(self):
        """Set the current unwrapped phase as the zero reference point"""
        if self.last_unwrapped_value is not None:
            # Set phase offset to the current unwrapped phase value
            self.phase_offset = self.last_unwrapped_value
        else:
            self.phase_offset = 0.0

    def save_distance_data(self):
        """Start saving distance data for the specified time length"""
        if not self.daq_running:
            self.show_message_box("Error", "DAQ must be running to save data.")
            return

        if self.is_saving:
            self.show_message_box("Error", "Already saving data. Please wait for current save to complete.")
            return

        try:
            self.save_time_length = float(self.save_time_input_tab3.text())
            self.downsample_points = int(self.downsample_input_tab3.text())
            if self.save_time_length <= 0:
                raise ValueError("Time length must be positive")
            if self.downsample_points <= 0:
                raise ValueError("Downsample points must be positive")
        except ValueError:
            self.show_message_box("Input Error",
                                  "Invalid inputs. Please enter positive numbers for time length and downsample points.")
            return

        # Start saving process
        self.is_saving = True
        self.save_start_time = len(self.distance_data) / self.sampling_rate if len(self.distance_data) > 0 else 0
        self.saved_distance_data = np.array([])
        self.saved_time_data = np.array([])

        # Update button text to show saving status
        self.save_data_button_tab3.setText(f'Saving... ({self.save_time_length}s)')
        self.save_data_button_tab3.setEnabled(False)

        # Show message to user
        self.show_message_box("Info", f"Started saving distance data for {self.save_time_length} seconds.")

    def finish_saving_data(self):
        """Finish saving data and open file dialog"""
        self.is_saving = False
        self.save_data_button_tab3.setText('Save Distance Data')
        self.save_data_button_tab3.setEnabled(True)

        if len(self.saved_distance_data) == 0:
            self.show_message_box("Error", "No data was collected during the save period.")
            return

        # Trim data to specified downsample points if necessary
        if len(self.saved_distance_data) > self.downsample_points:
            # Take evenly spaced samples to maintain time span
            indices = np.linspace(0, len(self.saved_distance_data) - 1, self.downsample_points, dtype=int)
            trimmed_distance_data = self.saved_distance_data[indices]
            trimmed_time_data = self.saved_time_data[indices]
        else:
            trimmed_distance_data = self.saved_distance_data
            trimmed_time_data = self.saved_time_data

        # Open file dialog to save data
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Distance Data",
            f"distance_data_{self.save_time_length}s.csv",
            "CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
        )

        if file_path:
            try:
                # Prepare data for saving
                data_to_save = np.column_stack((trimmed_time_data, trimmed_distance_data))

                # Save data with headers
                header = "Time (s),Distance (nm)"
                np.savetxt(file_path, data_to_save, delimiter=',', header=header, comments='', fmt='%.6f')

                # Update success message to show if data was trimmed
                data_info = f"Data points: {len(trimmed_distance_data)}"
                if len(self.saved_distance_data) > self.downsample_points:
                    data_info += f" (trimmed from {len(self.saved_distance_data)} using {self.downsample_points} points)"

                self.show_message_box("Success",
                                      f"Distance data saved successfully to:\n{file_path}\n\n{data_info}\nTime duration: {trimmed_time_data[-1]:.2f} seconds")

            except Exception as e:
                self.show_message_box("Error", f"Failed to save data: {str(e)}")

    def toggle_plot_visibility(self):
        """Toggle visibility of plots based on checkbox states"""
        self.daq_plot_widget_tab3_1.setVisible(self.show_ch1_checkbox.isChecked())
        self.daq_plot_widget_tab3_2.setVisible(self.show_ch2_checkbox.isChecked())
        self.xy_plot_widget.setVisible(self.show_xy_checkbox.isChecked())
        self.distance_plot_widget.setVisible(self.show_distance_checkbox.isChecked())
        self.residual_plot_widget.setVisible(self.show_residual_checkbox.isChecked())
        self.fft_plot_widget.setVisible(self.show_fft_checkbox.isChecked())

    @staticmethod
    def get_available_channels():
        channels = []
        try:
            system = System.local()
            for device in system.devices:
                ai_channels = [chan.name for chan in device.ai_physical_chans]
                channels.extend(ai_channels)
        except nidaqmx.DaqError as e:
            print(f"Failed to list available channels: {e}")
        return channels

    def toggle_daq_tab3(self):
        """Toggle between start and stop DAQ"""
        if not self.daq_running:
            self.start_daq_tab3()
        else:
            self.stop_daq_tab3()

    def start_daq_tab3(self):
        if not self.sampling_rate_input_tab3.text() or not self.samples_per_channel_input_tab3.text() or not self.samples_on_plot_input_tab3.text():
            self.show_message_box("Input Error",
                                  "Sampling rate, samples per channel, and samples on plot must not be empty.")
            return

        channels = [
            self.channel_select_dropdown_tab3_1.currentText(),
            self.channel_select_dropdown_tab3_2.currentText()
        ]

        try:
            self.sampling_rate = int(float(self.sampling_rate_input_tab3.text()))
        except ValueError:
            self.show_message_box("Input Error", "Invalid sampling rate. Please enter a valid number.")
            return

        try:
            samples_per_channel = int(float(self.samples_per_channel_input_tab3.text()))
        except ValueError:
            self.show_message_box("Input Error", "Invalid samples per channel. Please enter a valid number.")
            return

        try:
            self.samples_on_plot = int(float(self.samples_on_plot_input_tab3.text()))
        except ValueError:
            self.show_message_box("Input Error", "Invalid samples on plot. Please enter a valid number.")
            return

        self.buffer_size = self.samples_on_plot

        # Reset data buffers and phase unwrapping state
        self.daq_data_buffers = {1: np.array([]), 2: np.array([])}
        self.time_data_buffer = np.array([])
        self.processed_data_ch1 = np.array([])
        self.processed_data_ch2 = np.array([])
        self.distance_data = np.array([])
        self.residual_data = np.array([])
        self.fft_freq = np.array([])
        self.fft_magnitude = np.array([])
        self.last_unwrapped_value = None  # Reset phase unwrapping state

        self.daq_thread_tab3 = DAQThread(channels, self.sampling_rate, samples_per_channel)
        self.daq_thread_tab3.data_ready.connect(self.update_daq_plot_tab3)
        self.daq_thread_tab3.start()

        # Update button state
        self.daq_running = True
        self.toggle_daq_button_tab3.setText('Stop DAQ')

    def stop_daq_tab3(self):
        if self.daq_thread_tab3 is not None:
            self.daq_thread_tab3.stop()
            self.daq_thread_tab3.wait()
            self.daq_thread_tab3 = None

            # Update button state
            self.daq_running = False
            self.toggle_daq_button_tab3.setText('Start DAQ')

    def update_daq_plot_tab3(self, time_array, data_dict):
        if len(data_dict) == 0:
            self.show_message_box("DAQ Error", "An error occurred during data acquisition.")
            self.stop_daq_tab3()
            return

        # Update the plots for each channel
        for i, (channel, data) in enumerate(data_dict.items(), 1):
            self.daq_data_buffers[i] = np.concatenate((self.daq_data_buffers[i], data))

            # Trim the buffer to the specified size to avoid stacking
            if len(self.daq_data_buffers[i]) > self.buffer_size:
                self.daq_data_buffers[i] = self.daq_data_buffers[i][-self.buffer_size:]

            # Create a new time array for plotting
            self.time_data_buffer = np.linspace(0, len(self.daq_data_buffers[i]) / self.sampling_rate,
                                                len(self.daq_data_buffers[i]))

            # Update the DAQ data plot (original data) only if visible
            if i == 1 and self.show_ch1_checkbox.isChecked():
                self.daq_plot_widget_tab3_1.plot(self.time_data_buffer, self.daq_data_buffers[i], pen='#75DF75',
                                                 clear=True)
            elif i == 2 and self.show_ch2_checkbox.isChecked():
                self.daq_plot_widget_tab3_2.plot(self.time_data_buffer, self.daq_data_buffers[i], pen='#6D69BC',
                                                 clear=True)

        # Process data and update XY plot and distance plot
        if len(self.daq_data_buffers[1]) > 0 and len(self.daq_data_buffers[2]) > 0:
            # Process channel 1 data
            self.processed_data_ch1 = self.process_channel_data(self.daq_data_buffers[1])
            # Process channel 2 data
            self.processed_data_ch2 = self.process_channel_data(self.daq_data_buffers[2])

            # Ensure both processed channels have the same length for XY plotting
            min_length = min(len(self.processed_data_ch1), len(self.processed_data_ch2))
            if min_length > 0:
                ch1_processed = self.processed_data_ch1[-min_length:]
                ch2_processed = self.processed_data_ch2[-min_length:]
                phi_t = np.arctan2(ch1_processed, ch2_processed)

                # Use custom incremental unwrapping instead of np.unwrap
                phi_t_unwrapped = self.incremental_unwrap_array(phi_t)

                # Apply phase offset to start unwrapped phase from zero
                phi_t_unwrapped_zeroed = phi_t_unwrapped - self.phase_offset

                # Calculate distance using the formula: distance = (633.2 * phi_t_unwrapped_zeroed) / (8 * np.pi)
                distance = (633.2 * phi_t_unwrapped_zeroed) / (8 * np.pi)

                # Update distance buffer
                self.distance_data = np.concatenate((self.distance_data, distance))
                if len(self.distance_data) > self.buffer_size:
                    self.distance_data = self.distance_data[-self.buffer_size:]

                # Create time array for distance plot
                distance_time_array = np.linspace(0, len(self.distance_data) / self.sampling_rate,
                                                  len(self.distance_data))

                # Update XY plot only if visible
                if self.show_xy_checkbox.isChecked():
                    self.xy_plot_widget.plot(ch1_processed, ch2_processed, pen=None, symbol='o', symbolSize=3,
                                             symbolBrush='#FF6B6B', clear=True)

                # Update distance plot only if visible
                if self.show_distance_checkbox.isChecked():
                    self.distance_plot_widget.plot(distance_time_array, self.distance_data, pen='#0000FF',
                                                   clear=True)

                # Handle data saving if active
                if self.is_saving:
                    # Add current distance data to save buffer
                    self.saved_distance_data = np.concatenate((self.saved_distance_data, distance))

                    # Create corresponding time array for saved data
                    current_time = len(self.saved_distance_data) / self.sampling_rate
                    new_time_points = np.linspace(
                        len(self.saved_distance_data) - len(distance),
                        len(self.saved_distance_data) - 1,
                        len(distance)
                    ) / self.sampling_rate
                    self.saved_time_data = np.concatenate((self.saved_time_data, new_time_points))

                    # Check if we've collected enough data
                    if current_time >= self.save_time_length:
                        self.finish_saving_data()

                # Calculate and update residuals and FFT
                residuals, fft_freq, fft_magnitude = self.calculate_residuals_and_fft(self.distance_data,
                                                                                      distance_time_array)

                if len(residuals) > 0:
                    # Update residual plot only if visible
                    if self.show_residual_checkbox.isChecked():
                        self.residual_plot_widget.plot(distance_time_array, residuals, pen='#FF69B4', clear=True)

                    # Update FFT plot only if visible
                    if len(fft_freq) > 0 and len(fft_magnitude) > 0 and self.show_fft_checkbox.isChecked():
                        self.fft_plot_widget.plot(fft_freq, fft_magnitude, pen='#00CED1', clear=True)

    def clear_plot_tab3(self):
        self.daq_plot_widget_tab3_1.clear()
        self.daq_plot_widget_tab3_2.clear()
        self.xy_plot_widget.clear()
        self.distance_plot_widget.clear()
        self.residual_plot_widget.clear()
        self.fft_plot_widget.clear()
        self.daq_data_buffers = {1: np.array([]), 2: np.array([])}
        self.time_data_buffer = np.array([])
        self.processed_data_ch1 = np.array([])
        self.processed_data_ch2 = np.array([])
        self.distance_data = np.array([])
        self.residual_data = np.array([])
        self.fft_freq = np.array([])
        self.fft_magnitude = np.array([])
        self.last_unwrapped_value = None  # Reset phase unwrapping state
        self.phase_offset = 0.0  # Reset phase offset

    @staticmethod
    def show_message_box(title, message):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply white background with black text
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255))  # White background
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))  # Black text
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(255, 255, 255))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))  # Blue highlight
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # Increase font size and button size, set font to Arial
    font = QFont("Arial", 13)
    app.setFont(font)

    ex = SerialCommTest()
    ex.setStyleSheet("""
        QPushButton {
            font-size: 23px;
            padding: 20px;
            background-color: #2d2d2d;
            border: 3px solid #5a5a5a;
            border-radius: 10px;
        }
        QPushButton {
                background-color: white;
                color: black;
            }
            QLineEdit {
                background-color: white;
                color: black;
            }
            QLabel {
                color: black;
            }
            QCheckBox {
                color: black;
            }
            QComboBox {
                background-color: white;
                color: black;
            }
        QCheckBox {
            font-size: 20px;
            color: black;
            spacing: 10px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
        }
        QCheckBox::indicator:unchecked {
            background-color: #3a3a3a;
            border: 2px solid #5a5a5a;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            background-color: #42a5f5;
            border: 2px solid #42a5f5;
            border-radius: 3px;
        }
        QGroupBox {
            font-size: 23px;
            font-weight: bold;
            color: white;
            border: 3px solid #5a5a5a;
            border-radius: 10px;
            margin-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 5px;
            background-color: #2d2d2d;
        }
    """)
    ex.show()

    sys.exit(app.exec_())