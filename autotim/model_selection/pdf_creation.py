"""Class for storing metrics reports in a PDF"""
import os
from io import BytesIO

from fpdf import FPDF, XPos, YPos
from matplotlib import pyplot as plt


class MetricsReportPDF(FPDF):
    def __init__(self, metrics, model_name="best_model", **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        self.model_name = model_name

    def _get_width(self):
        available_width = self.w - 2 * self.l_margin
        return available_width

    def header(self):
        # render logos
        pdf_width = self.w - 40
        self.image(os.path.join(os.path.dirname(__file__), "report_data/servicemeister.jpg"), x=20, y=10, w=30)
        self.image(os.path.join(os.path.dirname(__file__), "report_data/inovex_logo.png"), x=pdf_width, y=10, w=20)

    def set_general_components(self):
        # add inovex font and color
        self.add_font("atkinson", "", os.path.join(os.path.dirname(__file__), "Atkinson_Hyperlegible/AtkinsonHyperlegible-Regular.ttf"), uni=True)
        self.add_font("atkinson", "B", os.path.join(os.path.dirname(__file__), "Atkinson_Hyperlegible/AtkinsonHyperlegible-Bold.ttf"), uni=True)
        self.add_font("atkinson", "I", os.path.join(os.path.dirname(__file__), "Atkinson_Hyperlegible/AtkinsonHyperlegible-Italic.ttf"), uni=True)
        self.set_text_color(5, 28, 89)

        # page setup
        self.add_page()
        self.set_margin(20)
        self.set_right_margin(40)
        self.set_font("atkinson", "B", 20)

        # add heading
        available_width = self._get_width()
        self.cell(w=available_width, h=50, txt=f"Performance-Bericht für Modell {self.model_name}", new_x=XPos.LEFT,
                  new_y=YPos.NEXT,
                  align="C")

    def create_table_of_content(self):
        self.set_font("atkinson", "B", 12)

        available_width = self._get_width()
        self.cell(w=available_width, h=10, txt="Inhalt", new_x=XPos.LEFT, new_y=YPos.NEXT, align="L")

        self.set_font("atkinson", "", 10)
        self.cell(h=10, txt="1.    Performance Metrics", new_x=XPos.LEFT, new_y=YPos.NEXT, align="L")
        self.cell(w=available_width, h=5, txt="2.    Confusion Matrix", new_x=XPos.LEFT, new_y=YPos.NEXT,
                  align="L")

    def create_metrics_table(self):
        self.set_text_color(5, 28, 89)
        self.set_font("atkinson", "B", 12)

        available_width = self._get_width()
        half_available_width = available_width / 2
        # add empty cell block
        self.cell(w=available_width, h=20, new_x=XPos.LEFT, new_y=YPos.NEXT)
        self.cell(w=(available_width / 2) - 20, h=20, txt="Performance Metrics", new_x=XPos.LEFT, new_y=YPos.NEXT,
                  align="C")

        self.image(os.path.join(os.path.dirname(__file__), "report_data/speedometer.png"), x=half_available_width / 2, w=20)

        self.set_text_color(0, 0, 0)
        self.set_font("atkinson", "", 10)

        # add header to metrics dictionary
        updated_metrics = {"Metric": "Value"}
        updated_metrics.update(self.metrics)

        self.set_right_margin(20)

        # create table
        with self.table(
                width=80,
                align="RIGHT",
                cell_fill_color=(244, 247, 255),
                cell_fill_mode="ROWS",
                col_widths=(30, 20),
                text_align=("LEFT", "RIGHT"),
        ) as table:
            self.set_y(120)
            for metric, value in updated_metrics.items():
                row = table.row()
                row.cell(metric.replace("_", " "))

                if isinstance(value, str):
                    row.cell(value)
                else:
                    row.cell(f"{value:.2f}")

    def integrate_confusion_matrix(self, confusion_matrix_fig):
        self.set_text_color(5, 28, 89)
        self.set_font("atkinson", "B", 12)

        available_width = self._get_width()
        # add empty cell block
        self.cell(w=available_width, h=30, new_x=XPos.LEFT, new_y=YPos.NEXT)

        half_available_width = available_width / 2
        self.cell(w=half_available_width - 20, h=20, txt="Confusion Matrix", border=0, new_x=XPos.LEFT, new_y=YPos.NEXT,
                  align="C")
        self.image(os.path.join(os.path.dirname(__file__), "report_data/evaluation.png"), x=half_available_width / 2, w=20)

        fig = confusion_matrix_fig
        img_buf = BytesIO()
        plt.savefig(img_buf, format="svg")
        self.image(img_buf, x=half_available_width + 27, y=180, w=75)
        img_buf.close()
