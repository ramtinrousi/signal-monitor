
import io
import os
import tempfile
import flet as ft
import pandas as pd
import numpy as np

# ====== رفع مشکل matplotlib روی اندروید ======
def get_writable_dir():
    """پیدا کردن یه مسیر قابل نوشتن برای هر پلتفرم"""
    candidates = [
        os.path.join(tempfile.gettempdir(), "mpl_config"),
        "/tmp/mpl_config",
        os.path.join(os.getcwd(), "mpl_config"),
        os.path.join(os.path.expanduser("~"), ".mpl_config"),
    ]
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            # تست نوشتن
            test_file = os.path.join(candidate, ".writetest")
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return candidate
        except (PermissionError, OSError, Exception):
            continue
    return None

config_dir = get_writable_dir()
if config_dir:
    os.environ['MPLCONFIGDIR'] = config_dir
    matplotlibrc_path = os.path.join(config_dir, 'matplotlibrc')
    try:
        if not os.path.exists(matplotlibrc_path):
            with open(matplotlibrc_path, 'w') as f:
                f.write('backend: Agg\n')
        os.environ['MATPLOTLIBRC'] = matplotlibrc_path
    except Exception:
        pass
# ============================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
import base64
import platform

if platform.system() == "Windows":
    import tkinter as tk
    from tkinter import filedialog
plt.style.use('dark_background')

class SignalMonitorApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "📊 Signal Monitor - Flet"
        self.page.window.width = 850
        self.page.window.height = 390
        self.page.window.resizable = True
        self.page.bgcolor = "#05070A"
        self.page.padding = 3
        self.page.spacing = 2

        self.all_groups_data = []
        self.current_group_index = 0
        self.zoom_start = 0
        self.zoom_end = 0
        self.total_samples = 0
        self.range_start = 0
        self.range_end = 0
        self.range_mode = 'sample'
        self.freq_value = 0.0
        self.voltage_range = 0.9
        self.axis_mode = 'Voltage'
        self.v_unit = 'Volt (V)'
        self.t_unit = 'Second (s)'

        self.fig, self.ax = plt.subplots(figsize=(5.0, 2.5))
        self.fig.patch.set_facecolor('#05070A')
        self.file_picker = ft.FilePicker()
        self.file_picker.on_result = self.on_file_picker_result
        self.page.overlay.append(self.file_picker)

        self.build_ui()

    def build_ui(self):
        self.main_info_label = ft.Text("No Data", size=9, color="#808c9d", weight=ft.FontWeight.BOLD)

        self.chart_image = ft.Image(
            src="",
            expand=True,
            fit="fill"
        )

        left_panel = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("📊 Signal Monitor", size=11, weight=ft.FontWeight.BOLD, color="#2456B2"),
                    self.main_info_label
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                self.chart_image
            ], spacing=1, expand=True),
            bgcolor="#0d121a",
            border_radius=4,
            padding=3,
            expand=71
        )

        self.load_btn = ft.ElevatedButton(
            "📂 Browse & Upload CSV",
            on_click=self.pick_file,
            bgcolor="#2456B2",
            color="#dcdfe4",
            height=22,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=3),
                text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD),
                padding=2
            )
        )

        self.group_dropdown = ft.Dropdown(
            label="Group",
            options=[ft.dropdown.Option("Select Group")],
            value="Select Group",
            text_size=9,
            dense=True,
            on_select=self.on_group_selected,
            expand=True
        )
        self.save_btn = ft.ElevatedButton(
            "Save", 
            on_click=self.save_group_to_txt, 
            bgcolor="#121721", 
            color="#ffd103", 
            height=22, 
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=3),
                side=ft.BorderSide(1, "#ffd103"),
                text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD),
                padding=2
            ), 
            expand=True
        )

        file_card = ft.Container(
            content=ft.Column([
                ft.Text("📁 File & Data-Set", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"),
                self.load_btn,
                ft.Row([self.group_dropdown, self.save_btn], spacing=2)
            ], spacing=1),
            bgcolor="#121721",
            border=ft.BorderSide(1, "#40516b"),
            border_radius=3,
            padding=3
        )

        # حالت ولتاژ/دیجیتال
        self.ymode_volt_btn = ft.ElevatedButton(
            "Volt", on_click=lambda e: self.set_ymode("Voltage"),
            bgcolor="#ffd103", color="#0d0d0d", height=21, expand=True,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0)
        )
        self.ymode_dig_btn = ft.ElevatedButton(
            "Digital", on_click=lambda e: self.set_ymode("Digital"),
            bgcolor="#1a2230", color="#dcdfe4", height=21, expand=True,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0)
        )
        self.ymode_group = ft.Row([ft.Text("Mode:", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"), self.ymode_volt_btn, self.ymode_dig_btn], spacing=2)

        # واحدهای ولتاژ
        self.v_v_btn = ft.ElevatedButton("V", on_click=lambda e: self.set_vunit("Volt (V)"), bgcolor="#ffd103", color="#0d0d0d", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        self.v_mv_btn = ft.ElevatedButton("mV", on_click=lambda e: self.set_vunit("Millivolt (mV)"), bgcolor="#1a2230", color="#dcdfe4", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        self.v_uv_btn = ft.ElevatedButton("uV", on_click=lambda e: self.set_vunit("Microvolt (uV)"), bgcolor="#1a2230", color="#dcdfe4", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        self.yunit_group = ft.Row([ft.Text("Unit:", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"), self.v_v_btn, self.v_mv_btn, self.v_uv_btn], spacing=2)

        self.vrange_text = ft.Text("0.9V", size=9, weight=ft.FontWeight.BOLD, color="#ffd103", text_align=ft.TextAlign.CENTER)
        
        vrange_controls = ft.Row([
            ft.ElevatedButton(
                content=ft.Text("−", size=12, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                on_click=self.decrease_voltage_range, bgcolor="#2456B2", color="#dcdfe4", width=28, height=21,
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), padding=0)
            ),
            self.vrange_text,
            ft.ElevatedButton(
                content=ft.Text("+", size=12, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER),
                on_click=self.increase_voltage_range, bgcolor="#ffd103", color="#0d0d0d", width=28, height=21,
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), padding=0)
            )
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        vertical_card = ft.Container(
            content=ft.Column([
                ft.Text("🔋 Vertical Setup", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"),
                self.ymode_group,
                self.yunit_group,
                vrange_controls
            ], spacing=1),
            bgcolor="#121721",
            border=ft.BorderSide(1, "#40516b"),
            border_radius=3,
            padding=3
        )

        self.freq_input = ft.TextField(hint_text="1000", text_size=9, height=21, content_padding=2)

        self.t_s_btn = ft.ElevatedButton("s", on_click=lambda e: self.set_tunit("Second (s)"), bgcolor="#ffd103", color="#0d0d0d", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        self.t_ms_btn = ft.ElevatedButton("ms", on_click=lambda e: self.set_tunit("Millisecond (ms)"), bgcolor="#1a2230", color="#dcdfe4", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        self.t_us_btn = ft.ElevatedButton("us", on_click=lambda e: self.set_tunit("Microsecond (us)"), bgcolor="#1a2230", color="#dcdfe4", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        tunit_row = ft.Row([ft.Text("T-Unit:", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"), self.t_s_btn, self.t_ms_btn, self.t_us_btn], spacing=2)

        self.rng_samp_btn = ft.ElevatedButton("Sample", on_click=lambda e: self.set_rangemode("Sample"), bgcolor="#ffd103", color="#0d0d0d", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        self.rng_time_btn = ft.ElevatedButton("Time", on_click=lambda e: self.set_rangemode("Time"), bgcolor="#1a2230", color="#dcdfe4", height=21, expand=True, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0))
        rangemode_row = ft.Row([ft.Text("Range:", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"), self.rng_samp_btn, self.rng_time_btn], spacing=2)

        # اصلاح طول فیلدهای S و E با استفاده از expand=True برای توازن کامل
        self.range_start_input = ft.TextField(hint_text="0", text_size=9, height=21, content_padding=2, expand=True)
        self.range_end_input = ft.TextField(hint_text="max", text_size=9, height=21, content_padding=2, expand=True)

        apply_btn = ft.ElevatedButton(
            "Apply", on_click=self.apply_range, bgcolor="#2456B2", color="#dcdfe4", height=21, expand=True,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0)
        )
        reset_btn = ft.ElevatedButton(
            "Reset", on_click=self.reset_view, bgcolor="#121721", color="#ffd103", height=21, expand=True,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=2), side=ft.BorderSide(1, "#ffd103"), text_style=ft.TextStyle(size=9, weight=ft.FontWeight.BOLD), padding=0)
        )

        horizontal_card = ft.Container(
            content=ft.Column([
                ft.Text("⏱️ Horizontal Setup", size=9, weight=ft.FontWeight.BOLD, color="#2456B2"),
                ft.Row([ft.Text("Freq(Hz):", size=9, color="#2456B2"), self.freq_input], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                tunit_row,
                rangemode_row,
                ft.Row([ft.Text("S:", size=9, color="#2456B2"), self.range_start_input, ft.Text("E:", size=9, color="#2456B2"), self.range_end_input], spacing=2),
                ft.Row([apply_btn, reset_btn], spacing=2)
            ], spacing=1),
            bgcolor="#121721",
            border=ft.BorderSide(1, "#40516b"),
            border_radius=3,
            padding=3
        )

        # پنل کنترل سمت راست کاملاً فشرده و متناسب بدون اسکرول
        right_panel = ft.Container(
            content=ft.Column([
                ft.Text("⚙️ Controls", size=10, weight=ft.FontWeight.BOLD, color="#2456B2"),
                file_card,
                vertical_card,
                horizontal_card
            ], spacing=2),
            bgcolor="#0d121a",
            border_radius=4,
            padding=3,
            expand=29
        )

        self.stat_samples = self.create_stat_box("Samples")
        self.stat_freq = self.create_stat_box("Frequency")
        self.stat_vmax = self.create_stat_box("V-Max")
        self.stat_vmin = self.create_stat_box("V-Min")
        self.stat_vp2p = self.create_stat_box("V-P2P")

        bottom_panel = ft.Container(
            content=ft.Row([
                self.stat_samples, self.stat_freq, self.stat_vmax, self.stat_vmin, self.stat_vp2p
            ], spacing=2, alignment=ft.MainAxisAlignment.SPACE_EVENLY),
            bgcolor="#0d121a",
            border=ft.BorderSide(1, "#40516b"),
            border_radius=4,
            padding=2
        )

        main_layout = ft.Column([
            ft.Row([left_panel, right_panel], spacing=2, expand=True),
            bottom_panel
        ], spacing=2, expand=True)

        self.page.add(main_layout)
        self.page.update()

    def create_stat_box(self, title):
        val_text = ft.Text("--", size=10, weight=ft.FontWeight.BOLD, color="#ffd103")
        container = ft.Container(
            content=ft.Column([
                ft.Text(title, size=8, weight=ft.FontWeight.BOLD, color="#2456B2", text_align=ft.TextAlign.CENTER),
                ft.Container(
                    content=val_text,
                    bgcolor="#05070A",
                    border_radius=2,
                    padding=1,
                    expand=True
                )
            ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor="#0d121a",
            border_radius=2,
            padding=1,
            expand=True
        )
        container.val_text = val_text
        return container

    def set_stat_value(self, box, val):
        box.val_text.value = str(val)
        box.val_text.update()

    def set_ymode(self, mode):
        self.axis_mode = mode
        if mode == "Voltage":
            self.ymode_volt_btn.bgcolor = "#ffd103"
            self.ymode_volt_btn.color = "#0d0d0d"
            self.ymode_dig_btn.bgcolor = "#1a2230"
            self.ymode_dig_btn.color = "#dcdfe4"
        else:
            self.ymode_dig_btn.bgcolor = "#ffd103"
            self.ymode_dig_btn.color = "#0d0d0d"
            self.ymode_volt_btn.bgcolor = "#1a2230"
            self.ymode_volt_btn.color = "#dcdfe4"
        self.ymode_volt_btn.update()
        self.ymode_dig_btn.update()
        if self.all_groups_data:
            self.plot_current_group()

    def set_vunit(self, unit):
        self.v_unit = unit
        for btn, u in [(self.v_v_btn, "Volt (V)"), (self.v_mv_btn, "Millivolt (mV)"), (self.v_uv_btn, "Microvolt (uV)")]:
            if u == unit:
                btn.bgcolor = "#ffd103"
                btn.color = "#0d0d0d"
            else:
                btn.bgcolor = "#1a2230"
                btn.color = "#dcdfe4"
            btn.update()
        if self.all_groups_data:
            self.plot_current_group()

    def set_tunit(self, unit):
        self.t_unit = unit
        for btn, u in [(self.t_s_btn, "Second (s)"), (self.t_ms_btn, "Millisecond (ms)"), (self.t_us_btn, "Microsecond (us)")]:
            if u == unit:
                btn.bgcolor = "#ffd103"
                btn.color = "#0d0d0d"
            else:
                btn.bgcolor = "#1a2230"
                btn.color = "#dcdfe4"
            btn.update()
        if self.all_groups_data:
            self.plot_current_group()

    def set_rangemode(self, mode):
        self.range_mode = mode.lower()
        if mode == "Sample":
            self.rng_samp_btn.bgcolor = "#ffd103"
            self.rng_samp_btn.color = "#0d0d0d"
            self.rng_time_btn.bgcolor = "#1a2230"
            self.rng_time_btn.color = "#dcdfe4"
        else:
            self.rng_time_btn.bgcolor = "#ffd103"
            self.rng_time_btn.color = "#0d0d0d"
            self.rng_samp_btn.bgcolor = "#1a2230"
            self.rng_samp_btn.color = "#dcdfe4"
        self.rng_samp_btn.update()
        self.rng_time_btn.update()
        self.range_start_input.value = ''
        self.range_end_input.value = ''
        self.page.update()

    def pick_file(self, e):
        if platform.system() == "Windows":
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="انتخاب فایل CSV", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.main_info_label.value = f"✅ File: {os.path.basename(file_path)}"
                self.page.update()
                self.load_csv_from_path(file_path)
            else:
                self.main_info_label.value = "❌ Cancelled"
                self.page.update()
        else:
            self.page.run_task(self._pick_files_async)

    async def _pick_files_async(self):
        try:
            await self.file_picker.pick_files(allowed_extensions=["csv"])
        except Exception as ex:
            self.main_info_label.value = f"❌ Error: {ex}"
            self.page.update()

    def on_file_picker_result(self, e):
        if e.files:
            try:
                self.load_csv_from_content(e.files[0].read())
            except Exception as ex:
                self.main_info_label.value = f"❌ Error: {ex}"
                self.page.update()

    def load_csv_from_content(self, content: bytes):
        try:
            self._process_dataframe(pd.read_csv(io.BytesIO(content), on_bad_lines='skip'))
        except Exception as ex:
            self.main_info_label.value = f"❌ Parse Error: {ex}"
            self.page.update()

    def load_csv_from_path(self, file_path):
        try:
            self._process_dataframe(pd.read_csv(file_path, on_bad_lines='skip'))
        except Exception as ex:
            self.main_info_label.value = f"❌ Parse Error: {ex}"
            self.page.update()

    def _process_dataframe(self, datas):
        filtered = datas[(datas['event_type'] == 'CharacteristicRead-Generic') &
                         (datas['characteristic_uuid'] == '15005991-b131-3396-014c-664c9867b917')].copy()
        filtered = filtered[filtered['value'].notna() & (filtered['value'] != '')]
        total = len(filtered)
        if total == 0:
            self.main_info_label.value = "❌ No Data"
            self.page.update()
            return

        filtered['time'] = pd.to_datetime(filtered['time'], format='%a %b %d %H:%M:%S GMT%z %Y', errors='coerce')
        filtered = filtered.sort_values('time').reset_index(drop=True)
        filtered['time_diff'] = filtered['time'].diff().dt.total_seconds()
        break_points = filtered[filtered['time_diff'] > 5].index.tolist()
        groups = []
        start = 0
        for bp in break_points:
            groups.append(filtered.iloc[start:bp])
            start = bp
        groups.append(filtered.iloc[start:])

        def process_hex_list(hex_list):
            bit_chars = []
            for hex_str in hex_list:
                if not isinstance(hex_str, str):
                    continue
                for byte_hex in hex_str.split():
                    try:
                        val = int(byte_hex, 16)
                        bit_chars.append(f"{val:08b}")
                    except ValueError:
                        pass
            full_bitstr = "".join(bit_chars)
            num_blocks = len(full_bitstr) // 10
            if num_blocks == 0:
                return np.array([], dtype=np.int16)
            return np.array([int(full_bitstr[i*10:(i+1)*10], 2) for i in range(num_blocks)], dtype=np.int16)

        self.all_groups_data = []
        group_options = []
        for idx, group_df in enumerate(groups):
            if len(group_df) == 0:
                continue
            decimal_vals = process_hex_list(group_df['value'].tolist())
            if len(decimal_vals) == 0:
                continue
            normalized_vals = (decimal_vals / 1023.0) * 0.9
            self.all_groups_data.append({
                'decimal': decimal_vals,
                'normalized': normalized_vals,
                'samples': len(decimal_vals)
            })
            group_options.append(ft.dropdown.Option(f"G{idx+1} ({len(decimal_vals)} pts)"))

        if not self.all_groups_data:
            self.main_info_label.value = "❌ No Samples"
            self.page.update()
            return

        self.group_dropdown.options = group_options
        self.group_dropdown.value = group_options[0].key
        self.main_info_label.value = f'✅ Loaded {total} rows'
        self.current_group_index = 0
        self.total_samples = self.all_groups_data[0]['samples']
        self.zoom_start = 0
        self.zoom_end = self.total_samples - 1
        self.range_start = 0
        self.range_end = self.total_samples - 1
        self.plot_current_group()

    def on_group_selected(self, e):
        val = self.group_dropdown.value
        if not val or val == 'Select Group' or not self.all_groups_data:
            return
        try:
            idx = int(val.split()[0][1:]) - 1
            if 0 <= idx < len(self.all_groups_data):
                self.current_group_index = idx
                self.total_samples = self.all_groups_data[idx]['samples']
                self.zoom_start = 0
                self.zoom_end = self.total_samples - 1
                self.range_start = 0
                self.range_end = self.total_samples - 1
                self.plot_current_group()
        except Exception:
            pass

    def increase_voltage_range(self, e):
        if self.axis_mode == 'Digital':
            return
        if self.voltage_range + 0.9 <= 3.61:
            self.voltage_range += 0.9
            self.vrange_text.value = f"{self.voltage_range:.1f}V"
            self.vrange_text.update()
            if self.all_groups_data:
                self.plot_current_group()

    def decrease_voltage_range(self, e):
        if self.axis_mode == 'Digital':
            return
        if self.voltage_range - 0.9 >= 0.89:
            self.voltage_range -= 0.9
            self.vrange_text.value = f"{self.voltage_range:.1f}V"
            self.vrange_text.update()
            if self.all_groups_data:
                self.plot_current_group()

    def apply_range(self, e):
        if not self.all_groups_data or self.total_samples == 0:
            return
        start_text = self.range_start_input.value
        end_text = self.range_end_input.value
        if not start_text or not end_text:
            return
        try:
            s, en = float(start_text), float(end_text)
            if self.range_mode == 'sample':
                si, ei = max(0, int(s)), min(self.total_samples - 1, int(en))
            else:
                try:
                    self.freq_value = float(self.freq_input.value)
                except ValueError:
                    self.freq_value = 0
                if self.freq_value <= 0:
                    self.main_info_label.value = "❌ Set Freq"
                    self.page.update()
                    return
                si = max(0, int(s * self.freq_value))
                ei = min(self.total_samples - 1, int(en * self.freq_value))

            if si >= ei:
                self.main_info_label.value = "❌ S < E"
                self.page.update()
                return
            self.range_start, self.range_end = si, ei
            self.zoom_start, self.zoom_end = si, ei
            self.plot_current_group()
        except Exception:
            self.main_info_label.value = "❌ Invalid"
            self.page.update()

    def reset_view(self, e):
        if self.total_samples == 0:
            return
        self.range_start = 0
        self.range_end = self.total_samples - 1
        self.range_start_input.value = ''
        self.range_end_input.value = ''
        self.zoom_start = 0
        self.zoom_end = self.total_samples - 1
        self.plot_current_group()

    def save_group_to_txt(self, e):
        if not self.all_groups_data:
            self.main_info_label.value = "❌ No Data"
            self.page.update()
            return
        try:
            data = self.all_groups_data[self.current_group_index]
            scale_y = 1000.0 if self.v_unit == 'Millivolt (mV)' else (1000000.0 if self.v_unit == 'Microvolt (uV)' else 1.0)
            y_values = data['decimal'] if self.axis_mode == 'Digital' else data['normalized'] * scale_y

            start_idx = max(0, self.zoom_start)
            end_idx = min(len(y_values) - 1, self.zoom_end)
            visible_y_values = y_values[start_idx:end_idx + 1]
            filename = f"group_{self.current_group_index + 1}_data.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Group {self.current_group_index + 1} Data Values\n")
                for val in visible_y_values:
                    f.write(f"{int(val) if self.axis_mode == 'Digital' else f'{val:.4f}'}\n")
            self.main_info_label.value = f"✅ Saved"
            self.page.update()
        except Exception:
            self.main_info_label.value = "❌ Save Error"
            self.page.update()

    def update_bottom_stats(self, data):
        try:
            self.freq_value = float(self.freq_input.value)
        except ValueError:
            self.freq_value = 0.0

        samples_count = len(data['decimal'])
        self.set_stat_value(self.stat_samples, f"{samples_count} Pts")
        self.set_stat_value(self.stat_freq, f"{self.freq_value:.1f} Hz" if self.freq_value > 0 else "N/A")

        if self.axis_mode == 'Digital':
            v_max, v_min = np.max(data['decimal']), np.min(data['decimal'])
            v_p2p = v_max - v_min
            unit_str = ""
            self.set_stat_value(self.stat_vmax, f"{int(v_max)}{unit_str}")
            self.set_stat_value(self.stat_vmin, f"{int(v_min)}{unit_str}")
            self.set_stat_value(self.stat_vp2p, f"{int(v_p2p)}{unit_str}")
        else:
            scale_y = 1000.0 if self.v_unit == 'Millivolt (mV)' else (1000000.0 if self.v_unit == 'Microvolt (uV)' else 1.0)
            unit_str = " mV" if self.v_unit == 'Millivolt (mV)' else (" uV" if self.v_unit == 'Microvolt (uV)' else " V")
            v_vals = data['normalized'] * scale_y
            v_max, v_min = np.max(v_vals), np.min(v_vals)
            v_p2p = v_max - v_min
            self.set_stat_value(self.stat_vmax, f"{v_max:.2f}{unit_str}")
            self.set_stat_value(self.stat_vmin, f"{v_min:.2f}{unit_str}")
            self.set_stat_value(self.stat_vp2p, f"{v_p2p:.2f}{unit_str}")

    def plot_current_group(self):
        if not self.all_groups_data:
            return
        data = self.all_groups_data[self.current_group_index]

        try:
            self.freq_value = float(self.freq_input.value)
        except ValueError:
            self.freq_value = 0.0

        self.ax.clear()
        self.ax.set_facecolor('#05070A')

        if self.axis_mode == 'Digital':
            y_values = data['decimal']
            y_label = 'Digital Amplitude'
            y_lim = (0, 1050)
            title_suffix = 'Digital'
        else:
            scale_y = 1000.0 if self.v_unit == 'Millivolt (mV)' else (1000000.0 if self.v_unit == 'Microvolt (uV)' else 1.0)
            unit_str = 'mV' if self.v_unit == 'Millivolt (mV)' else ('uV' if self.v_unit == 'Microvolt (uV)' else 'V')
            y_values = data['normalized'] * scale_y
            y_label = f'Voltage ({unit_str})'
            y_lim = (0, self.voltage_range * scale_y)
            title_suffix = f'Voltage (0-{self.voltage_range * scale_y:.1f}{unit_str})'

        start_idx = max(0, self.zoom_start)
        end_idx = min(len(y_values) - 1, self.zoom_end)
        y_zoomed_orig = y_values[start_idx:end_idx + 1]

        if self.freq_value > 0:
            raw_seconds_orig = np.arange(start_idx, end_idx + 1) / self.freq_value
            x_orig = raw_seconds_orig * 1000.0 if self.t_unit == 'Millisecond (ms)' else (raw_seconds_orig * 1000000.0 if self.t_unit == 'Microsecond (us)' else raw_seconds_orig)
            x_label = 'Time (ms)' if self.t_unit == 'Millisecond (ms)' else ('Time (us)' if self.t_unit == 'Microsecond (us)' else 'Time (s)')
            title_time = f' (fs={self.freq_value:.0f}Hz)'
        else:
            x_orig = np.arange(start_idx, end_idx + 1)
            x_label = 'Sample Index'
            title_time = ''

        if len(y_zoomed_orig) > 3 and self.axis_mode != 'Digital':
            x_indices = np.arange(len(y_zoomed_orig))
            x_dense_indices = np.linspace(0, len(y_zoomed_orig) - 1, len(y_zoomed_orig) * 10)
            cs = CubicSpline(x_indices, y_zoomed_orig)
            y_smooth = cs(x_dense_indices)

            if self.freq_value > 0:
                raw_seconds_dense = (start_idx + x_dense_indices) / self.freq_value
                x_smooth = raw_seconds_dense * 1000.0 if self.t_unit == 'Millisecond (ms)' else (raw_seconds_dense * 1000000.0 if self.t_unit == 'Microsecond (us)' else raw_seconds_dense)
            else:
                x_smooth = start_idx + x_dense_indices

            self.ax.plot(x_smooth, y_smooth, linewidth=1.0, color='#3A86FF', alpha=0.9)
            self.ax.plot(x_orig, y_zoomed_orig, 'o', color='#FFD166', markersize=2.5)
        else:
            self.ax.plot(x_orig, y_zoomed_orig, '-o', linewidth=1.0, color='#3A86FF', markerfacecolor='#FFD166', markeredgecolor='#FFD166', markersize=2.5)

        title_text = f'Group {self.current_group_index + 1} - {title_suffix}{title_time} ({len(y_zoomed_orig)} pts)'
        self.ax.set_title(title_text, color='#E2E8F0', fontsize=8, fontweight='bold', pad=2)
        self.ax.set_xlabel(x_label, color='#64748B', fontsize=6)
        self.ax.set_ylabel(y_label, color='#64748B', fontsize=6)
        self.ax.grid(True, alpha=0.4, color='#475569', linestyle='-', linewidth=0.5)
        self.ax.set_ylim(y_lim)

        for spine in self.ax.spines.values():
            spine.set_color('#0F172A')
        self.ax.tick_params(colors='#64748B', labelsize=6)

        self.fig.tight_layout(pad=0.3)

        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100, facecolor=self.fig.get_facecolor(), edgecolor='none')
        buf.seek(0)

        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        self.chart_image.src = f"data:image/png;base64,{image_base64}"
        self.chart_image.update()

        self.main_info_label.value = f"📊 G{self.current_group_index + 1} | {self.axis_mode}"
        self.update_bottom_stats(data)
        self.page.update()

def main(page: ft.Page):
    SignalMonitorApp(page)

if __name__ == '__main__':
    try:
        ft.run(main)
    except TypeError:
        ft.app(target=main)
