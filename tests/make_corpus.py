"""Synthetic corpus mirroring the shape of the sample one (the real starter corpus wasn't provided)."""
import os, sys, io
from pathlib import Path
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
import openpyxl, docx

def pdf(path, lines):
    c = canvas.Canvas(str(path)); y = 800
    for l in lines:
        c.drawString(60, y, l); y -= 18
    c.showPage(); c.save()

def main(root):
    r = Path(root)
    for d in ["archive", "engineering", "logs", "planning", "specs", "support", "vendor"]:
        (r / d).mkdir(parents=True, exist_ok=True)
    # --- pdf: current + withdrawn revision
    pdf(r/"specs/tq40_datasheet_r2.pdf", ["Orrery TQ-40 Datasheet, Revision 2", "This revision supersedes the withdrawn r1 datasheet.",
        "Electrical characteristics", "Maximum junction temperature (Tj max): 94 C", "Minimum junction temperature: -20 C", "Supply voltage: 12 V"])
    pdf(r/"specs/tq40_datasheet_r1_WITHDRAWN.pdf", ["WITHDRAWN - superseded by revision 2", "Orrery TQ-40 Datasheet, Revision 1",
        "Maximum junction temperature (Tj max): 105 C", "Supply voltage: 12 V"])
    # --- docx with a table
    d = docx.Document(); d.add_heading("FY27 Roadmap", 1); d.add_paragraph("Planning document. Dates are targets.")
    t = d.add_table(rows=1, cols=3); t.rows[0].cells[0].text, t.rows[0].cells[1].text, t.rows[0].cells[2].text = "Product", "Milestone", "Quarter"
    for row in [("TQ-40", "General availability", "Q1 FY27"), ("TQ-60", "Customer sampling", "Q3 FY27"), ("TQ-60", "General availability", "Q1 FY28")]:
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = v
    d.save(r/"planning/roadmap_fy27.docx")
    # --- xlsx, two sheets
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Parts"
    ws.append(["Part number", "Description", "Product", "Field replaceable"])
    ws.append(["ORR-FAN-1108-A", "Fan assembly, single-rotor (legacy)", "TQ-20", "yes"])
    ws.append(["ORR-FAN-2214-B", "Fan assembly, dual-rotor", "TQ-40", "yes"])
    ws.append(["ORR-PSU-0930-C", "Power supply module", "TQ-40", "yes"])
    w2 = wb.create_sheet("Lead times"); w2.append(["Part number", "Lead time (weeks)"]); w2.append(["ORR-FAN-2214-B", 6])
    wb.save(r/"support/rma_parts.xlsx")
    # --- csv bug database: the fixing row does NOT share vocabulary with the question
    rows = ["ticket,summary,status,fixed_in"]
    for i in range(1800, 1900):
        rows.append(f"ORR-{i},Assorted defect number {i} in subsystem {i%7},closed,4.{i%4}.{i%3}")
    rows[48] = "ORR-1847,Cooling loop regulation regression at sustained load,closed,4.3.2"
    (r/"support/bug_database.csv").write_text("\n".join(rows) + "\n")
    # --- text + log + python
    (r/"engineering/meridian_release_notes.txt").write_text("Meridian firmware release notes\n4.3.1: fan curve tuning. Fixes ORR-1790.\n4.3.2: stability release.\n")
    (r/"logs/prod_inference_2026-09-02.log").write_text(
        "2026-09-02T10:00:01 INFO inference started batch=44\n2026-09-02T10:14:22 WARN E7731 THERMAL_THROTTLE engaged on node 7\n"
        "2026-09-02T10:14:22 WARN known issue ORR-1847 (see tracker)\n2026-09-02T10:20:00 INFO recovered\n")
    (r/"engineering/ingest_service.py").write_text('"""Ingest service."""\n# batch timeout used to be 60 seconds\nBATCH_TIMEOUT_S = 180\nMAX_RETRIES = 3\n')
    # --- images (text only readable by a vision model)
    im = Image.new("RGB", (600, 300), "white"); dr = ImageDraw.Draw(im)
    for i, l in enumerate(["Backplane connector J1", "B12  PWR_GOOD", "B14  THERM_ALERT#", "B15  FAN_TACH"]): dr.text((20, 20+40*i), l, fill="black")
    im.save(r/"specs/backplane_pinout.png")
    im = Image.new("RGB", (600, 300), "white"); dr = ImageDraw.Draw(im)
    for i, l in enumerate(["Orrery Systems", "MODEL TQ-40", "BOARD REVISION REV-C2", "SN OS4-118823-77"]): dr.text((20, 20+40*i), l, fill="black")
    im.save(r/"support/asset_label.jpg")
    # --- the four hostile things
    pdf(r/"vendor/_plain.pdf", ["Supplier agreement", "Unit price of the TQ-40 at 10,000 units: 6412"])
    w = PdfWriter(); [w.add_page(p) for p in PdfReader(str(r/"vendor/_plain.pdf")).pages]; w.encrypt("s3cret")
    with open(r/"vendor/supplier_agreement_ENCRYPTED.pdf", "wb") as f: w.write(f)
    os.remove(r/"vendor/_plain.pdf")
    (r/"vendor/telemetry_capture.dat").write_bytes(bytes(range(256)) * 8)
    (r/"vendor/internal_audit.txt").write_text("Unit price of the TQ-40 at 10,000 units: 6412\n"); os.chmod(r/"vendor/internal_audit.txt", 0)

if __name__ == "__main__":
    main(sys.argv[1])
