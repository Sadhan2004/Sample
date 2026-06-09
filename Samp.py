import asyncio
import os
import json
import random
import uuid
from datetime import date
from openai import AsyncAzureOpenAI
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from PIL import Image, ImageDraw

# ==========================================
# 1. AZURE OPENAI CONFIGURATION
# ==========================================
# Replace these with your actual Azure credentials
AZURE_OPENAI_API_KEY = "YOUR_AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT" # e.g., "https://api.uhg.com/api/cloud/..."
OPENAI_API_VERSION = "2025-01-01-preview"
DEPLOYMENT_NAME = "gpt-4.1_2025-04-14"

client = AsyncAzureOpenAI(
    api_version=OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=DEPLOYMENT_NAME,
)

# ==========================================
# 2. FOLDER SETUP
# ==========================================
BASE_OUTPUT_DIR = "synthetic_lmn_data"
PDF_DIR = os.path.join(BASE_OUTPUT_DIR, "pdf")
IMG_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
LBL_DIR = os.path.join(BASE_OUTPUT_DIR, "labels")

def setup_directories():
    """Creates the necessary folder structure for outputs."""
    for directory in [BASE_OUTPUT_DIR, PDF_DIR, IMG_DIR, LBL_DIR]:
        os.makedirs(directory, exist_ok=True)
    print(f"Output directories verified/created at: {BASE_OUTPUT_DIR}/")

# ==========================================
# 3. LLM TEXT GENERATION (ASYNC)
# ==========================================
async def generate_lmn_text(validity=True):
    """Calls Azure OpenAI to generate the textual content of the LMN."""
    
    prompt = """
    Generate a detailed narrative for a "Letter of Medical Necessity".
    Return the output strictly as a JSON object with the following keys:
    - letterhead_name (String: realistic clinic/hospital name)
    - doctor_name (String: e.g., Dr. First Last)
    - issue_date (YYYY-MM-DD: within the last 6 months)
    - patient_name (String)
    - requested_item (String: include a made-up HCPCS code)
    - diagnosis (String: include a made-up ICD-10 code)
    - medical_necessity_statement (String: 2-3 paragraphs explaining the clinical need)
    - validity_period (String)

    Follow these rules based on your system instructions:
    - Make up realistic names, addresses, and medical conditions.
    """

    if validity:
        system_instruction = "You are a detailed medical administrative assistant creating VALID documents. For 'validity_period', you MUST use a valid phrase like 'Valid for 12 months' or 'Annual validity'."
    else:
        system_instruction = """You are a detailed medical administrative assistant creating INVALID documents. 
        Choose ONE of the following invalid scenarios to apply:
        1. For 'validity_period', use an invalid phrase like 'one-time use', 'single use', or 'valid for 3 months'.
        2. Make the 'medical_necessity_statement' incredibly vague so the link between diagnosis and item is completely unclear.
        3. Explicitly state in the narrative that 'No signature is required' and only type the doctor's name."""
        
    try:
        response = await client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            response_format={ "type": "json_object" } # Forces JSON output if supported by model
        )
        lmn_data = json.loads(response.choices[0].message.content)
        lmn_data['is_valid'] = validity 
        return lmn_data
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return None

# ==========================================
# 4. DOCUMENT & IMAGE RENDERING
# ==========================================
def create_random_scribble_signature(output_path, width=200, height=50):
    """Generates an image of a randomized scribble signature."""
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    num_points = random.randint(15, 30)
    points = []
    current_x = 10
    for _ in range(num_points):
        current_x += random.randint(5, 15)
        if current_x > width - 10: break
        y = random.randint(10, height - 10)
        points.append((current_x, y))
        
    if len(points) > 2:
        for i in range(len(points) - 1):
            line_width = random.randint(1, 3)
            draw.line([points[i], points[i+1]], fill="black", width=line_width)
            
    img.save(output_path)
    return output_path

def render_lmn_document(lmn_data):
    """Generates the PDF, Signature Image, and JSON Label in respective folders."""
    unique_id = uuid.uuid4().hex[:8]
    
    pdf_filename = f"lmn_{unique_id}.pdf"
    json_filename = f"lmn_{unique_id}.json"
    sig_filename = f"sig_{unique_id}.png"
    
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    json_path = os.path.join(LBL_DIR, json_filename)
    sig_path = os.path.join(IMG_DIR, sig_filename)
    
    # --- 1. Generate PDF ---
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.5*inch, height - 0.75*inch, lmn_data.get('letterhead_name', 'Medical Clinic'))
    c.setFont("Helvetica", 10)
    c.drawString(0.5*inch, height - 0.9*inch, "123 Medical Drive, Health City, ST 12345")
    c.drawString(0.5*inch, height - 1.05*inch, "Phone: (555) 123-4567")
    c.line(0.5*inch, height - 1.25*inch, width - 0.5*inch, height - 1.25*inch)
    
    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 1.6*inch, "LETTER OF MEDICAL NECESSITY")
    
    # Body Data
    c.setFont("Helvetica-Bold", 10)
    c.drawString(0.5*inch, height - 2.0*inch, f"Date: {lmn_data.get('issue_date', 'N/A')}")
    c.drawString(0.5*inch, height - 2.15*inch, f"Patient Name: {lmn_data.get('patient_name', 'N/A')}")
    c.drawString(0.5*inch, height - 2.30*inch, f"Diagnosis: {lmn_data.get('diagnosis', 'N/A')}")
    c.drawString(0.5*inch, height - 2.45*inch, f"Requested Item: {lmn_data.get('requested_item', 'N/A')}")
    c.drawString(0.5*inch, height - 2.60*inch, f"Validity Period: {lmn_data.get('validity_period', 'N/A')}")
    
    # Narrative
    styles = getSampleStyleSheet()
    body_style = styles["Normal"]
    body_style.fontName = "Helvetica"
    body_style.fontSize = 10
    narrative_style = ParagraphStyle('Narrative', parent=body_style, leading=14, spaceBefore=12)
    
    p = Paragraph(lmn_data.get('medical_necessity_statement', ''), narrative_style)
    w, h = p.wrap(width - 1*inch, height - 4*inch)
    p.drawOn(c, 0.5*inch, height - 4.5*inch - h)

    # Signature Block
    signature_y_start = height - 5.5*inch - h
    c.setFont("Helvetica", 10)
    c.drawString(0.5*inch, signature_y_start, "Sincerely,")
    
    # Determine if a signature image should be drawn based on invalidation rules
    needs_signature = True
    if not lmn_data['is_valid']:
        narrative_lower = lmn_data.get('medical_necessity_statement', '').lower()
        if "no signature" in narrative_lower or "no signature" in lmn_data.get('validity_period', '').lower():
            needs_signature = False
            c.drawString(0.5*inch, signature_y_start - 0.4*inch, "(Electronic Signature - Typed Name Only)")

    if needs_signature:
        create_random_scribble_signature(sig_path)
        c.drawImage(sig_path, 0.5*inch, signature_y_start - 0.6*inch, width=1.5*inch, height=0.4*inch, mask='auto')

    c.drawString(0.5*inch, signature_y_start - 0.8*inch, lmn_data.get('doctor_name', 'Doctor'))
    c.drawString(0.5*inch, signature_y_start - 0.95*inch, "Treating Physician")
    c.save()
    
    # --- 2. Generate Label (JSON) ---
    label_status = "Valid" if lmn_data['is_valid'] else "Invalid"
    reason = "Meets criteria for Valid LMN." if lmn_data['is_valid'] else "Failed LMN validation rules."
    
    # Basic logic to extract specific invalid reason for the label
    if not lmn_data['is_valid']:
        val_period = lmn_data.get('validity_period', '').lower()
        if "one-time" in val_period or "single" in val_period or "months" in val_period and "12" not in val_period:
             reason = "Invalid: Forbidden Validity Period indicator."
        elif not needs_signature:
             reason = "Invalid: Missing handwritten signature (Typed name only)."
        else:
             reason = "Invalid: Narrative lacks clear clinical justification."

    label_data = {
        "document_id": unique_id,
        "pdf_file": pdf_filename,
        "signature_image": sig_filename if needs_signature else None,
        "label": label_status,
        "reason": reason,
        "extracted_data": {
            "patient_name": lmn_data.get('patient_name'),
            "doctor_name": lmn_data.get('doctor_name'),
            "diagnosis": lmn_data.get('diagnosis'),
            "item_requested": lmn_data.get('requested_item'),
            "validity_period": lmn_data.get('validity_period')
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(label_data, f, indent=4)
        
    print(f"Generated: {pdf_filename} | Label: {label_status}")

# ==========================================
# 5. MAIN ASYNC BATCH EXECUTION
# ==========================================
async def main(total_documents, batch_size=5):
    setup_directories()
    
    tasks = []
    total_generated = 0
    
    print(f"Starting generation of {total_documents} documents in batches of {batch_size}...")
    
    while total_generated < total_documents:
        current_batch_size = min(batch_size, total_documents - total_generated)
        
        # 1. Queue up API calls
        for _ in range(current_batch_size):
            is_valid = random.choice([True, False]) # Mix of valid and invalid
            tasks.append(generate_lmn_text(validity=is_valid))
            
        # 2. Execute Batch
        results = await asyncio.gather(*tasks)
        
        # 3. Process Results locally (PDF and JSON creation)
        for data in results:
            if data:
                render_lmn_document(data)
                total_generated += 1
                
        tasks = [] # Clear tasks for next batch
        
        # Brief pause to respect API rate limits
        if total_generated < total_documents:
            await asyncio.sleep(1.5) 

    print(f"\n✅ Successfully generated {total_generated} documents!")
    print(f"Check the '{BASE_OUTPUT_DIR}' folder for outputs.")

# ==========================================
# 6. RUN SCRIPT
# ==========================================
if __name__ == "__main__":
    # Specify how many total documents you want to generate.
    # Start with a small number (e.g., 10) to test before running 1000+
    DOCUMENTS_TO_GENERATE = 10 
    
    # Adjust batch size depending on your Azure OpenAI Rate Limits (Tokens Per Minute)
    BATCH_SIZE = 5 
    
    asyncio.run(main(DOCUMENTS_TO_GENERATE, BATCH_SIZE))
