import pathlib

import cairosvg
import customtkinter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from screeninfo import get_monitors

from bbn import BBN
from ctktable import *
from logger import logger

customtkinter.set_appearance_mode(
    "Light"
)  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme(
    "blue"
)  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self, bbn: BBN):
        super().__init__()

        self.bbn_dataframe = bbn.get_bbn_dataframe()
        self.bbn_dataframe = self.bbn_dataframe.rename_axis(index="Requirement")
        # configure window
        self.title(f"PyBBN Assurance Case")
        # Get the screen width and height
        primary_monitor = get_monitors()[0]
        screen_width, screen_height = primary_monitor.width, primary_monitor.height

        # Calculate the x and y coordinates for the window to be centered
        x = (screen_width - self.winfo_reqwidth()) // 8
        y = (screen_height - self.winfo_reqheight()) // 8
        # Set the window's position
        self.geometry(f"+{x//2}+{y//2}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.image_path = bbn.get_assurance_case_png()
        self.image = Image.open(bbn.get_assurance_case_png())
        self.large_test_image = customtkinter.CTkImage(
            self.image, size=(self.image.width, self.image.height)
        )

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="PyBBN",
            font=customtkinter.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(
            self.sidebar_frame, command=self.save_data, text="Save Data"
        )
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(
            self.sidebar_frame,
            command=self.show_assurance_case,
            text="Show Assurance Case",
        )
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(
            self.sidebar_frame, command=self.show_beliefs, text="Show Beliefs"
        )
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:", anchor="w"
        )
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="UI Scaling:", anchor="w"
        )
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling_event,
        )
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        # self.entry = customtkinter.CTkEntry(self, placeholder_text="CTkEntry")
        # self.entry.grid(
        #     row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew"
        # )

        # self.main_button_1 = customtkinter.CTkButton(
        #     master=self,
        #     fg_color="transparent",
        #     border_width=2,
        #     text_color=("gray10", "#DCE4EE"),
        # )
        # self.main_button_1.grid(
        #     row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew"
        # )

        # create textbox
        self.textbox = customtkinter.CTkLabel(
            self,
            image=self.large_test_image,
            text="",
            width=self.image.width,
            height=self.image.height,
        )
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create tabview
        # self.tabview = customtkinter.CTkTabview(self, width=250)
        # self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        # self.tabview.add("CTkTabview")
        # self.tabview.add("Tab 2")
        # self.tabview.add("Tab 3")
        # self.tabview.tab("CTkTabview").grid_columnconfigure(
        #     0, weight=1
        # )  # configure grid of individual tabs
        # self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)

        # self.optionmenu_1 = customtkinter.CTkOptionMenu(
        #     self.tabview.tab("CTkTabview"),
        #     dynamic_resizing=False,
        #     values=["Value 1", "Value 2", "Value Long Long Long"],
        # )
        # self.optionmenu_1.grid(row=0, column=0, padx=20, pady=(20, 10))
        # self.combobox_1 = customtkinter.CTkComboBox(
        #     self.tabview.tab("CTkTabview"),
        #     values=["Value 1", "Value 2", "Value Long....."],
        # )
        # self.combobox_1.grid(row=1, column=0, padx=20, pady=(10, 10))
        # self.string_input_button = customtkinter.CTkButton(
        #     self.tabview.tab("CTkTabview"),
        #     text="Open CTkInputDialog",
        #     command=self.open_input_dialog_event,
        # )
        # self.string_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))
        # self.label_tab_2 = customtkinter.CTkLabel(
        #     self.tabview.tab("Tab 2"), text="CTkLabel on Tab 2"
        # )
        # self.label_tab_2.grid(row=0, column=0, padx=20, pady=20)

        bbn_table_values = self.bbn_dataframe.reset_index().values.tolist()
        # bbn_table_values = self.table_values
        frame = customtkinter.CTkFrame(self, fg_color="transparent")
        # frame.pack(expand=True, fill="both")
        frame.grid(row=1, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
        table = CTkTable(
            master=frame,
            row=len(bbn_table_values),
            column=len(bbn_table_values[0]),
            values=bbn_table_values,
            height=100,
            command=self.show,
        )
        table.grid(row=0, column=2, columnspan=1, padx=5, pady=5, sticky="")

        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkScrollableFrame(
            self, fg_color="transparent"
        )
        self.slider_progressbar_frame.grid(
            row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew"
        )
        self.slider_progressbar_frame.grid_columnconfigure(1, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        self.seg_button_1 = customtkinter.CTkSegmentedButton(
            self.slider_progressbar_frame
        )
        # self.seg_button_1.grid(
        #     row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew"
        # )
        # self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        # self.progressbar_1.grid(
        #     row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew"
        # )
        # self.progressbar_2 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        # self.progressbar_2.grid(
        #     row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew"
        # )

        # Create the label to display the slider value
        self.slider_value_label = customtkinter.CTkLabel(
            self.slider_progressbar_frame, text="0"
        )
        self.slider_value_label.grid(
            row=3, column=1, padx=(0, 0), pady=(0, 0), sticky="ew"
        )

        self.slider_1 = customtkinter.CTkSlider(
            self.slider_progressbar_frame,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.slider1,
        )
        self.slider_1.grid(row=3, column=0, padx=(0, 0), pady=(10, 10), sticky="ew")
        # self.slider_2 = customtkinter.CTkSlider(
        #     self.slider_progressbar_frame, orientation="vertical"
        # )
        # self.slider_2.grid(
        #     row=0, column=1, rowspan=5, padx=(10, 10), pady=(10, 10), sticky="ns"
        # )
        # self.progressbar_3 = customtkinter.CTkProgressBar(
        #     self.slider_progressbar_frame, orientation="vertical"
        # )
        # self.progressbar_3.grid(
        #     row=0, column=2, rowspan=5, padx=(10, 20), pady=(10, 10), sticky="ns"
        # )

        # create scrollable frame
        # self.scrollable_frame = customtkinter.CTkScrollableFrame(
        #     self, label_text="CTkScrollableFrame"
        # )
        # self.scrollable_frame.grid(
        #     row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew"
        # )
        # self.scrollable_frame.grid_columnconfigure(0, weight=1)
        # self.scrollable_frame_switches = []
        # for i in range(100):
        #     switch = customtkinter.CTkSwitch(
        #         master=self.scrollable_frame, text=f"CTkSwitch {i}"
        #     )
        #     switch.grid(row=i, column=0, padx=10, pady=(0, 20))
        #     self.scrollable_frame_switches.append(switch)

        # set default values
        # self.sidebar_button_3.configure(state="disabled", text="Disabled CTkButton")
        # self.checkbox_3.configure(state="disabled")
        # self.checkbox_1.select()
        # self.scrollable_frame_switches[0].select()
        # self.scrollable_frame_switches[4].select()
        # self.radio_button_3.configure(state="disabled")
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        # self.optionmenu_1.set("CTkOptionmenu")
        # self.combobox_1.set("CTkComboBox")
        # self.slider_1.configure(command=self.progressbar_2.set)
        # self.slider_2.configure(command=self.progressbar_3.set)
        # self.progressbar_1.configure(mode="indeterminnate")
        # self.progressbar_1.start()
        # self.textbox.insert("0.0", "CTkTextbox\n\n" + "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.\n\n" * 20)
        # self.seg_button_1.configure(values=["CTkSegmentedButton", "Value 2", "Value 3"])
        # self.seg_button_1.set("Value 2")

        # create a bar plot
        self.bar_plot_frame = customtkinter.CTkFrame(self)
        self.bar_plot_frame.grid(
            row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew"
        )

        fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
        # fig,ax = plt.subplots()

        # Extract data for each subplot
        missions = self.bbn_dataframe.loc["Meeting requirements"].tolist()
        collisions = self.bbn_dataframe.loc["Robot Collision under Threshold"].tolist()
        waypoints = self.bbn_dataframe.loc["Robot Nav Terrain under Threshold"].tolist()
        poses = self.bbn_dataframe.loc["Robot Pose under Threshold"].tolist()

        # # Plot each subplot
        self.plot_subplot(axes[0], missions, "P(Meeting requirements)")
        self.plot_subplot(axes[1], collisions, "P(Correct waypoints)")
        self.plot_subplot(axes[2], waypoints, "P(No collision)")
        self.plot_subplot(axes[3], poses, "P(Pose<=Region)")

        # Embed the matplotlib plot in the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self.bar_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

    # Plotting
    def plot_subplot(self, ax, df_subset, title):
        # Bar width and x positions
        bar_width = 0.35
        x_pos = np.arange(len(["True", "False"]))
        true_values, false_values = df_subset[0], df_subset[1]
        ax.bar(x_pos, true_values, bar_width, color="g", label="True")
        ax.bar(x_pos + bar_width, false_values, bar_width, color="r", label="False")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xticks([0, bar_width])
        ax.set_xticklabels(["True", "False"])

    def show(self, cell):
        logger.debug(
            f"Table value clicked: {cell['value']} ({cell['row']}x{cell['column']})"
        )

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(
            text="Type in a number:", title="CTkInputDialog"
        )
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def slider1(self, value):
        print("slider1", value)
        value = "{:.2f}".format(round(value, 2))
        self.slider_value_label.configure(text=f"{value}")

    def save_data(self):
        print("save_data click")

    def show_assurance_case(self):
        print("show_assurance_case click")

    def show_beliefs(self):
        print("show_beliefs click")


if __name__ == "__main__":
    app = App()
    app.mainloop()
