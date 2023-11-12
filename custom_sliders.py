import customtkinter

from bbn import BBN
from logger import logger


class ScrollableSliderFrame(customtkinter.CTkScrollableFrame):
    def __init__(
        self,
        master,
        probability_item_list,
        threshold_item_list,
        n_experiments,
        bbn: BBN,
        command=None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.n_experiments = n_experiments
        self.command = command
        self.slider_list = []
        self.slider_label_list = []
        self.slider_value_label_list = []
        self.slider_value_labels = {}
        self.bbn = bbn
        for i, item in enumerate(probability_item_list):
            self.add_probability_item(item)

        for i, item in enumerate(threshold_item_list):
            self.add_threshold_item(item)

    def add_probability_item(self, item):
        slider = customtkinter.CTkSlider(
            self,
            from_=0,
            to=1,
            number_of_steps=100,
        )
        slider.set(float("{:.2f}".format(round(item.probability_of_success, 2))))
        for key, value in self.bbn.assurance_case_dictionary.items():
            if value.get("text") == item.name:
                val = "{:.2f}".format(round(slider.get(), 2))
                value_label = customtkinter.CTkLabel(
                    self, text=f"{val}", compound="left", padx=5, anchor="w"
                )
                value_label.grid(
                    row=len(self.slider_value_label_list),
                    column=1,
                    pady=(0, 10),
                    sticky="w",
                )
                self.slider_value_label_list.append(value_label)
                self.slider_value_labels[key] = value_label
                slider.name = key
                label = customtkinter.CTkLabel(
                    self, text=f"{key}", compound="left", padx=5, anchor="w"
                )
                label.grid(
                    row=len(self.slider_label_list), column=2, pady=(0, 10), sticky="w"
                )

                self.slider_label_list.append(label)

        slider.configure(
            command=lambda value, slider=slider: self.combined_callback(value, slider)
        )
        # if self.command is None:
        #     slider.configure(
        #         command=lambda value, slider=slider: self.slider_callback( slider, value)
        #     )
        # else:
        #     slider.configure(
        #         command=lambda value, slider=slider: self.command(slider, value)
        #     )
        slider.grid(row=len(self.slider_list), column=0, padx=(0, 0), pady=(10, 10))

        self.slider_list.append(slider)

    def add_threshold_item(self, item):
        slider = customtkinter.CTkSlider(
            self,
            from_=0,
            to=self.n_experiments,
            number_of_steps=self.n_experiments,
        )
        slider.set(item.threshold)
        for key, value in self.bbn.assurance_case_dictionary.items():
            if value.get("text") == item.name:
                val = slider.get()
                value_label = customtkinter.CTkLabel(
                    self, text=f"{slider.get()}", compound="left", padx=5, anchor="w"
                )
                value_label.grid(
                    row=len(self.slider_value_label_list),
                    column=1,
                    pady=(0, 10),
                    sticky="w",
                )
                self.slider_value_label_list.append(value_label)
                self.slider_value_labels[key] = value_label
                slider.name = key
                label = customtkinter.CTkLabel(
                    self, text=f"{key}", compound="left", padx=5, anchor="w"
                )
                label.grid(
                    row=len(self.slider_label_list), column=2, pady=(0, 10), sticky="w"
                )
                self.slider_label_list.append(label)

        slider.configure(
            command=lambda value, slider=slider: self.combined_callback(value, slider)
        )
        # if self.command is None:
        #     slider.configure(
        #         command=lambda value, slider=slider: self.slider_callback(slider, value)
        #     )
        # else:
        #     slider.configure(
        #         command=lambda value, slider=slider: self.command(slider, value)
        #     )
        slider.grid(row=len(self.slider_list), column=0, padx=(0, 0), pady=(10, 10))

        self.slider_list.append(slider)

    def remove_item(self, item):
        for slider in self.slider_list:
            if item == slider.cget("text"):
                slider.destroy()
                self.slider_list.remove(slider)
                return

    def slider_callback(self, value, slider):
        name = slider.name
        value = "{:.2f}".format(round(value, 2))
        widget_name = slider.winfo_name()  # Get the widget name

    def slider_value_callback(self, value, slider):
        name = slider.name
        value = "{:.2f}".format(round(value, 2))
        self.slider_value_labels[name].configure(text=f"{value}")
        # handle plot/bbn evidence updates and new dataframes

    def combined_callback(self, value, slider):
        self.slider_callback(value, slider)
        self.slider_value_callback(value, slider)
        self.command(value, slider)
