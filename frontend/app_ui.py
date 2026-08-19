import shutil
import tempfile

import gradio as gr
import pandas as pd
import requests

BACKEND_URL = "http://localhost:8000/generate-emails"


def run_agent(csv_file, sender_company, sender_role):
    if csv_file is None:
        return "Upload a CSV first.", None, None
    if not sender_company.strip() or not sender_role.strip():
        return "Enter your company name and role first.", None, None

    # Backend expects a path on disk, so copy the upload somewhere stable
    tmp_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    shutil.copy(csv_file.name, tmp_path)

    try:
        resp = requests.post(
            BACKEND_URL,
            json={"csv_path": tmp_path, "sender_company": sender_company, "sender_role": sender_role},
            timeout=600,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Backend error: {e}", None, None

    data = resp.json()
    if "error" in data:
        return f"Error: {data['error']}", None, None

    summary = (
        f"Total leads: {data['total_leads']}  |  "
        f"Approved: {data['approved']}  |  Flagged: {data['flagged']}"
    )

    outbox_df = pd.DataFrame(data["outbox"])[
        ["company_name", "contact_name", "email", "subject", "body"]
    ] if data["outbox"] else pd.DataFrame(columns=["company_name", "contact_name", "email", "subject", "body"])

    flagged_df = pd.DataFrame(data["flagged_leads"])[
        ["company_name", "contact_name", "email", "revisions"]
    ] if data["flagged_leads"] else pd.DataFrame(columns=["company_name", "contact_name", "email", "revisions"])

    return summary, outbox_df, flagged_df


with gr.Blocks(title="Sales Outreach Agent") as demo:
    gr.Markdown("## Sales outreach agent\nUpload your leads CSV and generate personalized emails.")

    with gr.Row():
        company_input = gr.Textbox(label="Your company name", placeholder="e.g. upGrad")
        role_input = gr.Textbox(label="Your role", placeholder="e.g. Admissions Counsellor")

    csv_input = gr.File(label="Leads CSV", file_types=[".csv"])
    run_btn = gr.Button("Generate emails", variant="primary")

    summary_out = gr.Textbox(label="Summary", interactive=False)
    gr.Markdown("### Approved (outbox)")
    outbox_out = gr.Dataframe(wrap=True)
    gr.Markdown("### Flagged (needs manual review)")
    flagged_out = gr.Dataframe(wrap=True)

    run_btn.click(
        fn=run_agent,
        inputs=[csv_input, company_input, role_input],
        outputs=[summary_out, outbox_out, flagged_out],
    )

if __name__ == "__main__":
    demo.launch()