import fitz

pdf_path = "data/microsoft/SPD_Corporate_2026.pdf"

document = fitz.open(pdf_path)

print(len(document))

page_1 = document[0]

text = page_1.get_text()

print(text)