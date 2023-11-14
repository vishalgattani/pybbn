import pathlib
import textwrap
from tkinter import Canvas, PhotoImage, Scrollbar

import customtkinter
import matplotlib.pyplot as plt
import numpy as np
from customtkinter import CTkButton, CTkFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from screeninfo import get_monitors

from assurance_case import (
    n_experiments,
    p_correct_navigation,
    p_correct_pose,
    p_no_collision,
    sample_mission_bbn,
)
from bbn import BBN
from ctksliders import ScrollableSliderFrame
from ctktable import *
from doe import GoalNode, MaxThresholdNode, MinThresholdNode, SuccessNode, ThresholdNode
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
        self.plot_font_size = 5
        self.configure_gui(bbn)

    def update_gui_with_new_bbn(self, new_bbn: BBN):
        logger.warning(f"Updating BBN...")
        # Call this method when you want to update the GUI with a new BBN instance
        # self.scrollable_slider_frame.destroy()
        # self.table_frame.destroy()
        # self.sidebar_frame.destroy()
        # self.slider_progressbar_frame.destroy()
        # self.bar_plot_frame.destroy()
        self.configure_gui(new_bbn)

    def configure_gui(self, bbn: BBN):
        self.bbn = bbn
        self.n_experiments = bbn.n_experiments
        self.n_sliders = 0
        self.n_probability_sliders = 0
        self.n_threshold_sliders = 0
        self.probability_nodes = []
        self.threshold_nodes = []
        for id, node in self.bbn.nodes.items():
            if type(node).__name__ != GoalNode.__name__:
                self.n_sliders += 1
                if type(node).__name__ != SuccessNode.__name__:
                    self.n_threshold_sliders += 1
                    self.threshold_nodes.append(node)
            if type(node).__name__ == SuccessNode.__name__:
                self.n_probability_sliders += 1
                self.probability_nodes.append(node)

        self.bbn_dataframe = bbn.get_bbn_dataframe()
        self.bbn_dataframe = self.bbn_dataframe.rename_axis(index="Requirement")
        self.bbn_assurance_case_dictionary = bbn.assurance_case_dictionary
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
            self.image, size=(self.image.width // 1.2, self.image.height // 1.2)
        )

        # create textbox
        self.textbox = customtkinter.CTkLabel(
            self,
            image=self.large_test_image,
            text="",
            width=self.image.width,
            height=self.image.height,
        )
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create sidebarframe with widgets
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

        bbn_table_values = self.bbn_dataframe.reset_index().values.tolist()
        # bbn_table_values = self.table_values
        self.table_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        # self.table_frame.pack(expand=True, fill="both")
        self.table_frame.grid(
            row=1, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )
        self.table = CTkTable(
            master=self.table_frame,
            row=len(bbn_table_values),
            column=len(bbn_table_values[0]),
            values=bbn_table_values,
            height=100,
            command=self.show,
        )
        self.table.grid(row=0, column=2, columnspan=1, padx=5, pady=5, sticky="")

        # create slider and progressbarframe
        self.slider_progressbar_frame = customtkinter.CTkFrame(
            self,
            fg_color="transparent",
        )
        self.slider_progressbar_frame.grid(
            row=0, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew"
        )
        self.slider_progressbar_frame.grid_columnconfigure(1, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(self.n_sliders, weight=1)

        self.scrollable_slider_frame = ScrollableSliderFrame(
            master=self.slider_progressbar_frame,
            # height=self.winfo_reqheight(),
            probability_item_list=self.probability_nodes,
            threshold_item_list=self.threshold_nodes,
            n_experiments=self.n_experiments,
            bbn=bbn,
            command=self.handle_slider_value,
        )
        self.scrollable_slider_frame.grid(row=0, column=1, padx=0, pady=0, sticky="ns")
        self.scrollable_slider_frame.pack(fill="both", expand=True)

        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

        # create a bar plot
        self.bar_plot_frame = customtkinter.CTkFrame(self)
        self.bar_plot_frame.grid(
            row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew"
        )

        fig, axes = plt.subplots(1, 4, figsize=(5, 2), sharey=True)
        for i, idx in enumerate(self.bbn_dataframe.index.values.tolist()):
            self.plot_subplot(
                axes[i], self.bbn_dataframe.loc[idx].tolist(), f"P({idx})"
            )
        # Embed the matplotlib plot in the Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self.bar_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

    # Plotting
    def plot_subplot(self, ax, df_subset, title):
        title = "\n".join(textwrap.wrap(title, width=15))
        # Bar width and x positions
        bar_width = 0.35
        x_pos = np.arange(len(["True"]))
        true_values, false_values = df_subset[0], df_subset[1]
        ax.bar(x_pos, true_values, bar_width, color="g", label="True")
        ax.bar(x_pos + bar_width, false_values, bar_width, color="r", label="False")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="y", labelsize=self.plot_font_size)
        ax.set_title(title, fontsize=self.plot_font_size)
        ax.set_xticks([0, bar_width])
        ax.set_xticklabels(["True", "False"], fontsize=self.plot_font_size)
        # ax.set_xlabel(fontsize=self.plot_font_size)
        # ax.set_ylabel("",fontsize=self.plot_font_size)
        plt.tight_layout()

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
        ...
        # customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        # customtkinter.set_widget_scaling(new_scaling_float)

    def handle_slider_value(self, value, slider, sliderlist):
        value = "{:.2f}".format(round(value, 2))
        vals = []
        for s in sliderlist:
            vals.append(s.get())
        new_bbn = sample_mission_bbn(
            self.n_experiments, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]
        )
        self.update_gui_with_new_bbn(new_bbn=new_bbn)

    def save_data(self):
        print("save_data click")

    def show_assurance_case(self):
        print("show_assurance_case click")

    def show_beliefs(self):
        print("show_beliefs click")


if __name__ == "__main__":
    app = App()
    app.mainloop()
