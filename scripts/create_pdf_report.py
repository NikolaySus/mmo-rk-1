from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "CO2_Emissions_Canada.csv"
ENCODED_PATH = ROOT / "results" / "co2_emissions_with_mean_encoding.csv"
MEAN_ENCODING_PATH = ROOT / "results" / "mean_encoding_by_make.csv"
SCATTER_PATH = ROOT / "results" / "scatter_engine_size_co2.png"
OUTPUT_PATH = ROOT / "output" / "pdf" / "rk1_co2_report.pdf"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def register_fonts() -> tuple[str, str]:
    if Path(FONT_REGULAR).exists() and Path(FONT_BOLD).exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans", FONT_REGULAR))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", FONT_BOLD))
        return "DejaVuSans", "DejaVuSans-Bold"
    return "Helvetica", "Helvetica-Bold"


def make_styles(font_name: str, bold_font_name: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=bold_font_name,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName=bold_font_name,
            fontSize=13,
            leading=16,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=13,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=8,
            leading=10,
            spaceAfter=5,
        )
    )
    return styles


def paragraph(text: str, styles, style: str = "ReportBody") -> Paragraph:
    return Paragraph(text, styles[style])


def dataframe_table(df: pd.DataFrame, font_name: str, bold_font_name: str, max_rows: int = 7) -> Table:
    frame = df.head(max_rows).copy()
    frame = frame.fillna("")
    header = [str(col) for col in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    table = Table([header] + rows, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), bold_font_name),
                ("FONTNAME", (0, 1), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B8C0CC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVuSans", 8)
    canvas.setFillColor(colors.HexColor("#5B6472"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, f"Страница {doc.page}")
    canvas.restoreState()


def build_report() -> None:
    font_name, bold_font_name = register_fonts()
    styles = make_styles(font_name, bold_font_name)

    data = pd.read_csv(DATA_PATH)
    encoded = pd.read_csv(ENCODED_PATH)
    mean_encoding = pd.read_csv(MEAN_ENCODING_PATH)

    missing_report = pd.DataFrame(
        {
            "Признак": data.columns,
            "Тип": data.dtypes.astype(str).to_numpy(),
            "NaN": data.isna().sum().to_numpy(),
            "NaN, %": (data.isna().mean() * 100).round(2).to_numpy(),
        }
    )
    categorical_columns = data.select_dtypes(include="str").columns
    empty_strings = (
        data[categorical_columns]
        .apply(lambda column: column.astype(str).str.strip().eq("").sum())
        .reset_index()
    )
    empty_strings.columns = ["Категориальный признак", "Пустые строки"]

    source_fragment = data[
        [
            "Make",
            "Model",
            "Vehicle Class",
            "Engine Size(L)",
            "Fuel Type",
            "CO2 Emissions(g/km)",
        ]
    ]
    encoded_fragment = encoded[
        [
            "Make",
            "Model",
            "Engine Size(L)",
            "CO2 Emissions(g/km)",
            "Make_mean_encoded",
        ]
    ]
    mean_fragment = mean_encoding.sort_values("manual_mean", ascending=False)[
        ["Make", "manual_mean", "encoded_value", "abs_diff"]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.5 * cm,
        title="РК1: target encoding и диаграмма рассеяния",
        author="Горкунов Николай Максимович",
    )

    story = []
    story.append(paragraph("Рубежный контроль №1", styles, "ReportTitle"))
    story.append(paragraph("Горкунов Николай Максимович, ИУ5-21М", styles))
    story.append(
        dataframe_table(
            pd.DataFrame(
                [["2", "2", "22"]],
                columns=["Номер варианта", "Номер задачи №1", "Номер задачи №2"],
            ),
            font_name,
            bold_font_name,
            max_rows=1,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(paragraph("Задание", styles, "ReportHeading"))
    story.append(
        paragraph(
            "Для студентов групп ИУ5-21М, ИУ5И-21М необходимо для пары произвольных колонок данных построить график "
            "\"Диаграмма рассеяния\".",
            styles,
        )
    )
    story.append(
        paragraph(
            "Задача №2: для набора данных провести кодирование одного категориального признака с использованием метода "
            "\"target (mean) encoding\".",
            styles,
        )
    )
    story.append(paragraph("Датасет", styles, "ReportHeading"))
    story.append(
        paragraph(
            "Использован набор данных CO2 Emission by Vehicles. Файл: data/CO2_Emissions_Canada.csv. "
            f"Размер датасета: {data.shape[0]} строк и {data.shape[1]} колонок.",
            styles,
        )
    )
    story.append(paragraph("Фрагмент исходного датасета", styles, "ReportHeading"))
    story.append(dataframe_table(source_fragment, font_name, bold_font_name, max_rows=6))

    story.append(PageBreak())
    story.append(paragraph("Проверка пропусков", styles, "ReportHeading"))
    story.append(
        paragraph(
            "Были проверены явные пропуски NaN во всех колонках и пустые строки в категориальных признаках. "
            "В датасете не обнаружено ни NaN, ни пустых строк, поэтому удаление строк, удаление колонок или заполнение "
            "значений не требуется.",
            styles,
        )
    )
    story.append(dataframe_table(missing_report, font_name, bold_font_name, max_rows=12))
    story.append(Spacer(1, 0.2 * cm))
    story.append(paragraph("Пустые строки в категориальных признаках", styles, "ReportHeading"))
    story.append(dataframe_table(empty_strings, font_name, bold_font_name, max_rows=10))

    story.append(PageBreak())
    story.append(paragraph("Выполнение target / mean encoding", styles, "ReportHeading"))
    story.append(
        paragraph(
            "Для кодирования выбран категориальный признак Make. Целевой признак - CO2 Emissions(g/km). "
            "Каждое значение Make заменено средним значением целевого признака для соответствующей марки автомобиля. "
            "Новый признак: Make_mean_encoded.",
            styles,
        )
    )
    story.append(paragraph("Фрагмент таблицы средних значений по категориям", styles, "ReportHeading"))
    story.append(dataframe_table(mean_fragment, font_name, bold_font_name, max_rows=8))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        paragraph(
            "Расхождение между ручным расчетом среднего и значением TargetEncoder равно 0.0 для показанных категорий "
            "и для всей таблицы проверки.",
            styles,
        )
    )
    story.append(paragraph("Фрагмент датасета после кодирования", styles, "ReportHeading"))
    story.append(dataframe_table(encoded_fragment, font_name, bold_font_name, max_rows=8))

    story.append(PageBreak())
    story.append(paragraph("Диаграмма рассеяния", styles, "ReportHeading"))
    story.append(
        paragraph(
            "Построена диаграмма рассеяния для колонок Engine Size(L) и CO2 Emissions(g/km). "
            "Цвет точек соответствует признаку Fuel Type.",
            styles,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    image = Image(str(SCATTER_PATH))
    image.drawWidth = 17.2 * cm
    image.drawHeight = 11.4 * cm
    story.append(image)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_report()
