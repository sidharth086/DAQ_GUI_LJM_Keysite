import sys
import time
import csv
import os
import threading
import numpy as np
import labjack.ljm
import matplotlib.pyplot as plt
from collections import deque
from simple_pid import PID
import CoolProp.CoolProp as CP
import serial 

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QTextEdit,
                             QFileDialog, QCheckBox, QGridLayout, QLineEdit)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.gridspec import GridSpec

try:
    import pyvisa
except ImportError:
    pyvisa = None

class LabJackGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTV Control System - Water Fluid")
        self.setGeometry(100, 100, 1400, 950) 
        
        self.handle_pro = None
        self.handle_std = None
        self.rm = None
        self.keysight = None
        self.julabo = None
        
        self.flow_alert_active = False
        self.temp_alert_active = False
        self.is_reading = False 
        
        self.initialize_system_settings()
        self.initialize_data_structures()
        self.init_ui()
        
        # Note: connect_labjacks() is no longer called here automatically.
        # It is triggered by the UI button.
        self.start_threads()

    def initialize_system_settings(self):
        self.thermocouple_channels = [0, 2, 4, 10]
        self.temp_labels = {0: "T_In", 2: "T_Vap_Out", 4: "T_Liq_Out", 10: "T_Cond_Out"}
        self.pressure_channels = [6, 7, 8, 13]
        self.pressure_labels = {6: "P_In", 7: "P_Vap_Out", 8: "P_Liq_Out", 13: "P_Cond_Out"}
        self.surf_tc_channels = [101, 102, 103]
        self.surf_temp_labels = {101: "Surf_T1", 102: "Surf_T2", 103: "Surf_T3"}
        self.flow_meter_channel = 9
        self.PWM_DAC_CHANNEL = "DAC0"

        self.setpoint = 35.0
        self.Kp, self.Ki, self.Kd = 4.0, 0.02, 3.0
        self.pwm_frequency = 1.0
        self.current_duty_cycle = 0
        self.manual_mode = False
        self.pump_state = False
        self.refrigerant = "Water"
        self.csv_filename = "ttv_water_test_data.csv"
        
        self.initialize_csv()
        self.pid = PID(self.Kp, self.Ki, self.Kd, setpoint=self.setpoint)
        self.pid.output_limits = (0, 100)

    def initialize_csv(self):
        header = [
            "Timestamp", "T_In", "T_Vap_Out", "T_Liq_Out", "T_Cond_Out", 
            "P_In", "P_Vap_Out", "P_Liq_Out", "P_Cond_Out", 
            "Flow_LPM", "Duty_Cycle", "Surf_T1", "Surf_T2", "Surf_T3", "Julabo_Temp"
        ]
        if not os.path.isfile(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def initialize_data_structures(self):
        self.time_window = deque(maxlen=100)
        self.temp_data = {ch: deque(maxlen=100) for ch in self.thermocouple_channels}
        self.pressure_data = {ch: deque(maxlen=100) for ch in self.pressure_channels}
        self.flow_data = deque(maxlen=100)
        self.surf_temp_data = {ch: deque(maxlen=100) for ch in self.surf_tc_channels}
        self.running = False
        self.stop_flag = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_loop)

    def init_ui(self):
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #4a4a4a;
                border-radius: 5px;
                margin-top: 15px;
                background-color: #fcfcfc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ==========================================
        # LEFT PANEL: Controls & Status Log
        # ==========================================
        controls_layout = QVBoxLayout()
        
        # --- System Controls Group ---
        data_group = QGroupBox("System Controls")
        dg_lay = QVBoxLayout()
        self.start_btn = QPushButton("START ACQUISITION"); self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn = QPushButton("STOP"); self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self.stop_acquisition)
        self.save_btn = QPushButton("SAVE DATA AS..."); self.save_btn.clicked.connect(self.save_data_dialog)
        self.save_btn.setStyleSheet("background-color: #e1f5fe; font-weight: bold;")
        dg_lay.addWidget(self.start_btn); dg_lay.addWidget(self.stop_btn); dg_lay.addWidget(self.save_btn)
        data_group.setLayout(dg_lay); controls_layout.addWidget(data_group)

        # --- NEW: LabJack Connections Group ---
        lj_group = QGroupBox("LabJack Connections")
        lj_lay = QVBoxLayout()
        
        lj_pro_lay = QHBoxLayout()
        lj_pro_lay.addWidget(QLabel("T7-Pro (Main) SN/IP:"))
        self.lj_pro_input = QLineEdit("470035856")
        lj_pro_lay.addWidget(self.lj_pro_input)
        lj_lay.addLayout(lj_pro_lay)
        
        lj_std_lay = QHBoxLayout()
        lj_std_lay.addWidget(QLabel("T7-Std (Pump) SN/IP:"))
        self.lj_std_input = QLineEdit("470034208")
        lj_std_lay.addWidget(self.lj_std_input)
        lj_lay.addLayout(lj_std_lay)
        
        self.btn_connect_lj = QPushButton("Connect LabJacks")
        self.btn_connect_lj.clicked.connect(self.connect_labjacks)
        lj_lay.addWidget(self.btn_connect_lj)
        
        lj_group.setLayout(lj_lay)
        controls_layout.addWidget(lj_group)

        # --- Julabo Presto Control Group ---
        j_group = QGroupBox("Julabo Presto Control")
        j_lay = QVBoxLayout()
        
        j_conn_lay = QHBoxLayout()
        self.j_com_input = QLineEdit("COM3")
        self.btn_j_connect = QPushButton("Connect Serial")
        self.btn_j_connect.clicked.connect(self.connect_julabo)
        j_conn_lay.addWidget(self.j_com_input); j_conn_lay.addWidget(self.btn_j_connect)
        j_lay.addLayout(j_conn_lay)

        j_sp_lay = QHBoxLayout()
        j_sp_lay.addWidget(QLabel("Target (°C):"))
        self.j_sp_spin = QDoubleSpinBox()
        self.j_sp_spin.setRange(-50, 200)
        self.j_sp_spin.setValue(23.0)
        self.btn_j_set = QPushButton("Set Temp")
        self.btn_j_set.clicked.connect(self.set_julabo_temp)
        j_sp_lay.addWidget(self.j_sp_spin); j_sp_lay.addWidget(self.btn_j_set)
        j_lay.addLayout(j_sp_lay)

        j_ctrl_lay = QHBoxLayout()
        self.btn_j_start = QPushButton("START")
        self.btn_j_stop = QPushButton("STOP")
        self.btn_j_stop.setStyleSheet("background-color: #ef5350; color: white; font-weight: bold;") 
        self.btn_j_start.clicked.connect(lambda: self.set_julabo_run(True))
        self.btn_j_stop.clicked.connect(lambda: self.set_julabo_run(False))
        j_ctrl_lay.addWidget(self.btn_j_start); j_ctrl_lay.addWidget(self.btn_j_stop)
        j_lay.addLayout(j_ctrl_lay)

        self.lbl_j_temp = QLabel("Actual Fluid Temp: <b>N/A</b>")
        self.lbl_j_temp.setStyleSheet("font-size: 14px; color: #1565c0;")
        j_lay.addWidget(self.lbl_j_temp)
        j_group.setLayout(j_lay); controls_layout.addWidget(j_group)

        # --- Keysight DAQ Connection ---
        visa_group = QGroupBox("Keysight DAQ Connection")
        visa_lay = QVBoxLayout()
        visa_lay.addWidget(QLabel("VISA Address:"))
        self.visa_input = QLineEdit("USB0::0x2A8D::0x5101::MY58045827::0::INSTR")
        visa_lay.addWidget(self.visa_input)
        self.btn_connect_keysight = QPushButton("Connect Keysight")
        self.btn_connect_keysight.clicked.connect(self.connect_keysight)
        visa_lay.addWidget(self.btn_connect_keysight)
        visa_group.setLayout(visa_lay); controls_layout.addWidget(visa_group)

        # --- Pump Control ---
        p_group = QGroupBox("Pump Control"); p_lay = QHBoxLayout()
        self.pump_status_label = QLabel("Pump: OFF")
        
        self.btn_pump_on = QPushButton("ON")
        self.btn_pump_on.clicked.connect(lambda: self.set_pump(True))
        
        self.btn_pump_off = QPushButton("OFF")
        self.btn_pump_off.clicked.connect(lambda: self.set_pump(False))
        self.btn_pump_off.setStyleSheet("background-color: #ef5350; color: white; font-weight: bold;") 
        
        p_lay.addWidget(self.pump_status_label); p_lay.addWidget(self.btn_pump_on); p_lay.addWidget(self.btn_pump_off)
        p_group.setLayout(p_lay); controls_layout.addWidget(p_group)

        # --- PID Group ---
        pid_group = QGroupBox("Heater Control (PID)")
        pg_lay = QVBoxLayout()
        sp_lay = QHBoxLayout()
        sp_lay.addWidget(QLabel("<b>Setpoint (°C):</b>"))
        self.setpoint_spin = QDoubleSpinBox(); self.setpoint_spin.setRange(0, 100); self.setpoint_spin.setValue(self.setpoint)
        self.setpoint_spin.valueChanged.connect(self.update_setpoint)
        sp_lay.addWidget(self.setpoint_spin); pg_lay.addLayout(sp_lay)
        
        tunings_lay = QHBoxLayout()
        tunings_lay.addWidget(QLabel("Kp:"))
        self.kp_spin = QDoubleSpinBox(); self.kp_spin.setRange(0, 500); self.kp_spin.setSingleStep(0.5)
        self.kp_spin.setValue(self.Kp); self.kp_spin.valueChanged.connect(self.update_pid_tunings)
        tunings_lay.addWidget(self.kp_spin)
        
        tunings_lay.addWidget(QLabel("Ki:"))
        self.ki_spin = QDoubleSpinBox(); self.ki_spin.setRange(0, 500); self.ki_spin.setSingleStep(0.01)
        self.ki_spin.setDecimals(3)
        self.ki_spin.setValue(self.Ki); self.ki_spin.valueChanged.connect(self.update_pid_tunings)
        tunings_lay.addWidget(self.ki_spin)
        
        tunings_lay.addWidget(QLabel("Kd:"))
        self.kd_spin = QDoubleSpinBox(); self.kd_spin.setRange(0, 500); self.kd_spin.setSingleStep(0.5)
        self.kd_spin.setValue(self.Kd); self.kd_spin.valueChanged.connect(self.update_pid_tunings)
        tunings_lay.addWidget(self.kd_spin)
        pg_lay.addLayout(tunings_lay)

        manual_lay = QHBoxLayout()
        self.manual_check = QCheckBox("Manual PWM Control")
        self.manual_check.stateChanged.connect(self.toggle_manual)
        
        self.manual_duty_spin = QDoubleSpinBox()
        self.manual_duty_spin.setRange(0, 100)
        self.manual_duty_spin.setSuffix(" %")
        self.manual_duty_spin.setEnabled(False) 
        
        manual_lay.addWidget(self.manual_check)
        manual_lay.addWidget(QLabel("Duty Cycle:"))
        manual_lay.addWidget(self.manual_duty_spin)
        pg_lay.addLayout(manual_lay)
        
        pid_group.setLayout(pg_lay); controls_layout.addWidget(pid_group)

        # --- Status Log ---
        self.status_log = QTextEdit(); self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("border: 1px solid #ccc; border-radius: 4px; background-color: #fafafa;")
        controls_layout.addWidget(self.status_log)

        # ==========================================
        # RIGHT PANEL: Data Table & Plots
        # ==========================================
        display_panel = QVBoxLayout()
        
        stats_group = QGroupBox("Live Sensory Data"); grid = QGridLayout()
        headers = ["Location", "Temp (°C)", "Pressure (PSIA)", "Sat Temp (°C)", "Subcool (°C)"]
        for i, h in enumerate(headers): grid.addWidget(QLabel(f"<b>{h}</b>"), 0, i)
        
        self.lbl_in_t = QLabel("N/A"); self.lbl_in_p = QLabel("N/A"); self.lbl_in_sat = QLabel("N/A"); self.lbl_in_sub = QLabel("N/A")
        self.lbl_vap_t = QLabel("N/A"); self.lbl_vap_p = QLabel("N/A"); self.lbl_vap_sat = QLabel("N/A"); self.lbl_vap_sub = QLabel("N/A")
        self.lbl_liq_t = QLabel("N/A"); self.lbl_liq_p = QLabel("N/A"); self.lbl_liq_sat = QLabel("N/A"); self.lbl_liq_sub = QLabel("N/A")
        self.lbl_cond_t = QLabel("N/A"); self.lbl_cond_p = QLabel("N/A"); self.lbl_cond_sat = QLabel("N/A"); self.lbl_cond_sub = QLabel("N/A")
        self.lbl_flow = QLabel("N/A"); self.lbl_duty_cycle = QLabel("N/A")

        rows = [("Inlet", self.lbl_in_t, self.lbl_in_p, self.lbl_in_sat, self.lbl_in_sub),
                ("Vap Out", self.lbl_vap_t, self.lbl_vap_p, self.lbl_vap_sat, self.lbl_vap_sub),
                ("Liq Out", self.lbl_liq_t, self.lbl_liq_p, self.lbl_liq_sat, self.lbl_liq_sub),
                ("Cond Out", self.lbl_cond_t, self.lbl_cond_p, self.lbl_cond_sat, self.lbl_cond_sub)]
        
        for r_idx, (name, t, p, s, sc) in enumerate(rows, 1):
            grid.addWidget(QLabel(name), r_idx, 0)
            grid.addWidget(t, r_idx, 1); grid.addWidget(p, r_idx, 2); grid.addWidget(s, r_idx, 3); grid.addWidget(sc, r_idx, 4)
            
        grid.addWidget(QLabel("<b>Flow:</b>"), 5, 0); grid.addWidget(self.lbl_flow, 5, 1)
        grid.addWidget(QLabel("<b>Duty Cycle:</b>"), 5, 2); grid.addWidget(self.lbl_duty_cycle, 5, 3)
        stats_group.setLayout(grid); display_panel.addWidget(stats_group)

        self.fig_main = plt.figure(figsize=(10, 8), tight_layout=True)
        gs = GridSpec(2, 2, figure=self.fig_main)
        
        self.ax_t = self.fig_main.add_subplot(gs[0, 0])
        self.ax_surf = self.fig_main.add_subplot(gs[0, 1])
        self.ax_p = self.fig_main.add_subplot(gs[1, 0])
        self.ax_f = self.fig_main.add_subplot(gs[1, 1])
        
        self.canvas_main = FigureCanvas(self.fig_main)
        self.canvas_main.setStyleSheet("border: 1px solid #ccc;")
        display_panel.addWidget(self.canvas_main)

        main_layout.addLayout(controls_layout, 1) 
        main_layout.addLayout(display_panel, 4)   

    def save_data_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV Data", self.csv_filename, "CSV Files (*.csv)")
        if filename:
            self.csv_filename = filename
            self.status_log.append(f"📁 Saving data to: {filename}")
            self.initialize_csv()

    def connect_labjacks(self):
        self.btn_connect_lj.setText("Connecting...")
        QApplication.processEvents()
        
        pro_id = self.lj_pro_input.text().strip()
        std_id = self.lj_std_input.text().strip()
        
        success_pro = False
        success_std = False
        
        # Connect T7-Pro (Main)
        try:
            if pro_id:
                # "ANY" allows connection via USB or Ethernet/Wi-Fi automatically
                self.handle_pro = labjack.ljm.openS("T7", "ANY", pro_id)
                self.status_log.append(f"✅ T7-Pro Connected: {pro_id}")
                self.configure_sensors()
                success_pro = True
        except Exception as e:
            self.handle_pro = None
            self.status_log.append(f"❌ T7-Pro Error: {e}")

        # Connect T7-Std (Pump)
        try:
            if std_id:
                self.handle_std = labjack.ljm.openS("T7", "ANY", std_id)
                self.status_log.append(f"✅ T7-Std Connected: {std_id}")
                success_std = True
        except Exception as e:
            self.handle_std = None
            self.status_log.append(f"❌ T7-Std Error: {e}")

        # Update UI Button based on connection success
        if success_pro and success_std:
            self.btn_connect_lj.setText("Connected (Both)")
            self.btn_connect_lj.setStyleSheet("background-color: #c8e6c9; color: black; font-weight: bold;")
        elif success_pro or success_std:
            self.btn_connect_lj.setText("Partial Connection")
            self.btn_connect_lj.setStyleSheet("background-color: #fff59d; color: black; font-weight: bold;") # Yellow warning
        else:
            self.btn_connect_lj.setText("Retry Connections")
            self.btn_connect_lj.setStyleSheet("")

    def connect_julabo(self):
        port = self.j_com_input.text().strip()
        self.btn_j_connect.setText("Connecting...")
        QApplication.processEvents()
        try:
            self.julabo = serial.Serial(
                port=port, baudrate=4800, parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE, bytesize=serial.SEVENBITS, timeout=0.5
            )
            self.julabo.write(b"version\r")
            time.sleep(0.1)
            resp = self.julabo.read_all().decode('ascii').strip().replace('\x00', '')
            
            if resp:
                self.status_log.append(f"✅ Julabo Connected: {resp}")
                self.btn_j_connect.setText("Connected")
                self.btn_j_connect.setStyleSheet("background-color: #c8e6c9; color: black; font-weight: bold;")
            else:
                self.status_log.append("⚠️ Julabo port opened, but no response.")
                self.btn_j_connect.setText("Retry Connection")
        except Exception as e:
            self.julabo = None
            self.status_log.append(f"❌ Julabo Error: {e}")
            self.btn_j_connect.setText("Retry Connection")

    def set_julabo_temp(self):
        if self.julabo and self.julabo.is_open:
            sp = self.j_sp_spin.value()
            self.julabo.write(f"out_sp_00 {sp:.2f}\r".encode('ascii'))
            self.status_log.append(f"🌡️ Julabo Setpoint -> {sp:.2f} °C")
        else: self.status_log.append("❌ Connect Julabo first!")

    def set_julabo_run(self, state):
        if self.julabo and self.julabo.is_open:
            self.julabo.write(b"out_mode_05 1\r" if state else b"out_mode_05 0\r")
            self.status_log.append("▶️ Julabo Circulator Started" if state else "⏹️ Julabo Circulator Stopped")
            if state:
                self.btn_j_start.setStyleSheet("background-color: #66bb6a; color: white; font-weight: bold;") 
                self.btn_j_stop.setStyleSheet("") 
            else:
                self.btn_j_start.setStyleSheet("") 
                self.btn_j_stop.setStyleSheet("background-color: #ef5350; color: white; font-weight: bold;") 
        else: 
            self.status_log.append("❌ Connect Julabo first!")

    def connect_keysight(self):
        if not pyvisa: return self.status_log.append("⚠️ pyvisa not installed.")
        self.btn_connect_keysight.setText("Connecting...")
        QApplication.processEvents()
        try:
            if not self.rm: self.rm = pyvisa.ResourceManager()
            self.keysight = self.rm.open_resource(self.visa_input.text().strip())
            self.keysight.timeout = 5000 
            self.keysight.write("*RST"); self.keysight.write("*CLS"); time.sleep(1)
            
            self.keysight.write("CONF:TEMP TCOUPLE,T,(@101,102,103)")
            self.keysight.write("ROUT:SCAN (@101,102,103)")
            self.keysight.write("TRIG:SOUR IMM") 
            self.keysight.write("TRIG:COUN 1")   
            
            self.status_log.append(f"✅ Keysight Connected: {self.keysight.query('*IDN?').strip()}")
            self.btn_connect_keysight.setText("Connected")
            self.btn_connect_keysight.setStyleSheet("background-color: #c8e6c9; color: black; font-weight: bold;")
        except Exception as e:
            self.keysight = None
            self.status_log.append(f"❌ Keysight Error: {e}")
            self.btn_connect_keysight.setText("Retry Connection")

    def configure_sensors(self):
        for ain in self.thermocouple_channels:
            labjack.ljm.eWriteName(self.handle_pro, f"AIN{ain}_NEGATIVE_CH", ain + 1)
            labjack.ljm.eWriteName(self.handle_pro, f"AIN{ain}_EF_INDEX", 24)
            labjack.ljm.eWriteName(self.handle_pro, f"AIN{ain}_EF_CONFIG_A", 1)

    def update_pid_tunings(self):
        self.Kp = self.kp_spin.value(); self.Ki = self.ki_spin.value(); self.Kd = self.kd_spin.value()
        self.pid.tunings = (self.Kp, self.Ki, self.Kd)
        self.status_log.append(f"🔧 PID Updated: Kp={self.Kp:.1f}, Ki={self.Ki:.3f}, Kd={self.Kd:.1f}")

    def update_setpoint(self, val): 
        self.setpoint = val; self.pid.setpoint = val

    def toggle_manual(self, state): 
        self.manual_mode = (state == Qt.Checked)
        self.manual_duty_spin.setEnabled(self.manual_mode)
        if self.manual_mode:
            self.status_log.append("⚠️ Switched to Manual PWM Control.")
        else:
            self.status_log.append("🔄 Switched back to PID Control.")
    
    def set_pump(self, state):
        self.pump_state = state
        if self.handle_std: labjack.ljm.eWriteName(self.handle_std, "DAC0", 5.0 if state else 0.0)
        self.pump_status_label.setText(f"Pump: {'ON' if state else 'OFF'}")
        
        if state:
            self.btn_pump_on.setStyleSheet("background-color: #66bb6a; color: white; font-weight: bold;") 
            self.btn_pump_off.setStyleSheet("") 
        else:
            self.btn_pump_on.setStyleSheet("") 
            self.btn_pump_off.setStyleSheet("background-color: #ef5350; color: white; font-weight: bold;") 

    def start_acquisition(self):
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.time_window.clear()
        for ch in self.thermocouple_channels: self.temp_data[ch].clear()
        for ch in self.pressure_channels: self.pressure_data[ch].clear()
        self.flow_data.clear()
        for ch in self.surf_tc_channels: self.surf_temp_data[ch].clear()
        
        self.timer.start(1000) 
        self.status_log.append("▶️ Acquisition Started. Plots have been reset.")

    def stop_acquisition(self):
        self.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.timer.stop() 
        if self.handle_pro: 
            labjack.ljm.eWriteName(self.handle_pro, self.PWM_DAC_CHANNEL, 0)
        self.status_log.append("⏹️ Acquisition Paused/Stopped.")

    def update_loop(self):
        if not self.handle_pro or self.is_reading: 
            return
            
        self.is_reading = True
        
        try:
            ts = time.time()
            temps = {ch: labjack.ljm.eReadName(self.handle_pro, f"AIN{ch}_EF_READ_A") for ch in self.thermocouple_channels}
            press = {ch: (labjack.ljm.eReadName(self.handle_pro, f"AIN{ch}") * 20) for ch in self.pressure_channels}
            flow = labjack.ljm.eReadName(self.handle_pro, f"AIN{self.flow_meter_channel}") * 10
            
            surf = {101: np.nan, 102: np.nan, 103: np.nan}
            
            if self.keysight:
                try:
                    vals = self.keysight.query_ascii_values("READ?")
                    if len(vals) >= 3:
                        surf[101] = vals[0]; surf[102] = vals[1]; surf[103] = vals[2]
                except Exception as e: 
                    print(f"Keysight Error: {e}")
                    self.status_log.append(f"⚠️ Keysight Read Error! Clearing buffer...")
                    try:
                        self.keysight.clear()
                    except: 
                        pass
                
            j_temp_val = np.nan
            if self.julabo and self.julabo.is_open:
                try:
                    self.julabo.write(b"in_pv_00\r"); time.sleep(0.05) 
                    resp = self.julabo.read_until(b'\r').decode('ascii').strip().replace('\x00', '')
                    if resp:
                        j_temp_val = float(resp)
                        self.lbl_j_temp.setText(f"Actual Fluid Temp: <b>{j_temp_val:.2f} °C</b>")
                except Exception: pass

            if flow < 0.25:
                if not self.flow_alert_active:
                    self.status_log.append("⚠️ <span style='color:red;'><b>ALERT: Flow drop!</b></span>")
                    self.lbl_flow.setStyleSheet("color: red; font-weight: bold;")
                    self.flow_alert_active = True
            else:
                if self.flow_alert_active:
                    self.status_log.append("✅ <span style='color:green;'>Flow rate recovered.</span>")
                    self.lbl_flow.setStyleSheet("")
                    self.flow_alert_active = False

            if any(t > 110 for t in surf.values() if not np.isnan(t)):
                if not self.temp_alert_active:
                    self.status_log.append("🔥 <span style='color:red;'><b>ALERT: Surface Temp > 110 °C!</b></span>")
                    self.temp_alert_active = True
            else:
                if self.temp_alert_active:
                    self.status_log.append("✅ <span style='color:green;'>Surface temps normalized.</span>")
                    self.temp_alert_active = False

            if not self.manual_mode: 
                self.current_duty_cycle = self.pid(temps[0])
            else:
                self.current_duty_cycle = self.manual_duty_spin.value()

            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ts] + list(temps.values()) + list(press.values()) + 
                                [flow, self.current_duty_cycle] + list(surf.values()) + [j_temp_val])

            self.time_window.append(ts)
            for ch in self.thermocouple_channels: self.temp_data[ch].append(temps[ch])
            for ch in self.pressure_channels: self.pressure_data[ch].append(press[ch])
            self.flow_data.append(flow)
            for ch in self.surf_tc_channels: self.surf_temp_data[ch].append(surf[ch])

            self.update_plots(); self.update_ui_labels(temps, press, flow)
        except Exception as e: 
            print(f"Loop error: {e}")
        finally:
            self.is_reading = False

    def update_ui_labels(self, temps, press, flow):
        def f(v, d=1): return f"{v:.{d}f}" if not np.isnan(v) else "N/A"
        self.lbl_in_t.setText(f(temps[0])); self.lbl_in_p.setText(f(press[6]))
        self.lbl_vap_t.setText(f(temps[2])); self.lbl_vap_p.setText(f(press[7]))
        self.lbl_liq_t.setText(f(temps[4])); self.lbl_liq_p.setText(f(press[8]))
        self.lbl_cond_t.setText(f(temps[10])); self.lbl_cond_p.setText(f(press[13]))
        
        for p_ch, t_ch, lbl_s, lbl_sc in [(6,0,self.lbl_in_sat, self.lbl_in_sub), (7,2,self.lbl_vap_sat, self.lbl_vap_sub), 
                                          (8,4,self.lbl_liq_sat, self.lbl_liq_sub), (13,10,self.lbl_cond_sat, self.lbl_cond_sub)]:
            try:
                sat_t = CP.PropsSI("T", "P", press[p_ch] * 6894.76, "Q", 0, self.refrigerant) - 273.15
                lbl_s.setText(f(sat_t)); lbl_sc.setText(f(sat_t - temps[t_ch]))
            except: pass
        self.lbl_flow.setText(f"{f(flow, 2)} LPM"); self.lbl_duty_cycle.setText(f"{f(self.current_duty_cycle)} %")

    def update_plots(self):
        if not self.time_window: return
        t = [x - self.time_window[0] for x in self.time_window]
        
        self.ax_t.clear(); self.ax_p.clear(); self.ax_f.clear(); self.ax_surf.clear()
        for ax in [self.ax_t, self.ax_p, self.ax_f, self.ax_surf]:
            ax.tick_params(left=True, right=True, labelleft=True, labelright=True)
            ax.grid(True, linestyle='--', alpha=0.6)
        
        for ch in self.thermocouple_channels: self.ax_t.plot(t, self.temp_data[ch], label=self.temp_labels[ch])
        self.ax_t.set_title("System Temperatures (°C)")
        self.ax_t.set_ylabel("Temp (°C)")
        self.ax_t.legend(loc='upper left')
        
        for ch in self.surf_tc_channels: self.ax_surf.plot(t, self.surf_temp_data[ch], label=self.surf_temp_labels[ch])
        self.ax_surf.set_title("Surface Temperatures (°C)")
        self.ax_surf.set_ylabel("Temp (°C)")
        self.ax_surf.legend(loc='upper left')

        for ch in self.pressure_channels: self.ax_p.plot(t, self.pressure_data[ch], label=self.pressure_labels[ch])
        self.ax_p.set_title("System Pressures (PSIA)")
        self.ax_p.set_ylabel("Pressure (PSIA)")
        self.ax_p.legend(loc='upper left')
        
        self.ax_f.plot(t, self.flow_data, color='green')
        self.ax_f.set_title("Coolant Flow Rate (LPM)")
        self.ax_f.set_ylabel("Flow (LPM)")
        
        self.canvas_main.draw()

    def start_threads(self):
        self.pwm_thread = threading.Thread(target=self.pwm_logic, daemon=True); self.pwm_thread.start()

    def pwm_logic(self):
        while not self.stop_flag:
            if self.handle_pro:
                duty = self.current_duty_cycle / 100.0
                if duty > 0: labjack.ljm.eWriteName(self.handle_pro, self.PWM_DAC_CHANNEL, 5.0); time.sleep(duty)
                if duty < 1.0: labjack.ljm.eWriteName(self.handle_pro, self.PWM_DAC_CHANNEL, 0.0); time.sleep(1.0 - duty)
            else: time.sleep(1)

    def closeEvent(self, event):
        self.stop_flag = True
        if self.handle_pro: 
            try: labjack.ljm.eWriteName(self.handle_pro, self.PWM_DAC_CHANNEL, 0)
            except: pass
        if self.keysight:
            try: self.keysight.close()
            except: pass
        if self.rm:
            try: self.rm.close()
            except: pass
        if self.julabo and self.julabo.is_open:
            try:
                self.julabo.write(b"out_mode_05 0\r"); time.sleep(0.1); self.julabo.close()
            except: pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); gui = LabJackGUI(); gui.show(); sys.exit(app.exec_())