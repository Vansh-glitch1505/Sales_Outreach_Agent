import os

import gradio as gr
import pandas as pd
import requests


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000/generate-emails"
)


def run_agent(csv_file, sender_company, sender_role):

    if csv_file is None:
        return "Upload a CSV first.", None, None

    if not sender_company.strip() or not sender_role.strip():
        return (
            "Enter your company name and role first.",
            None,
            None
        )

    try:

        with open(csv_file.name, "rb") as f:

            files = {
                "file": (
                    os.path.basename(csv_file.name),
                    f,
                    "text/csv"
                )
            }

            form_data = {
                "sender_company": sender_company,
                "sender_role": sender_role
            }

            response = requests.post(
                BACKEND_URL,
                files=files,
                data=form_data,
                timeout=600
            )

        response.raise_for_status()

    except requests.RequestException as e:

        return (
            f"Backend error: {e}",
            None,
            None
        )

    try:

        data = response.json()

    except Exception:

        return (
            "Backend returned an invalid response.",
            None,
            None
        )

    if "error" in data:

        return (
            f"Error: {data['error']}",
            None,
            None
        )

    summary = (
        f"Total leads: {data['total_leads']}  |  "
        f"Approved: {data['approved']}  |  "
        f"Flagged: {data['flagged']}"
    )

    if data["outbox"]:

        outbox_df = pd.DataFrame(
            data["outbox"]
        )[
            [
                "contact_name",
                "email",
                "subject",
                "body"
            ]
        ]

    else:

        outbox_df = pd.DataFrame(
            columns=[
                "contact_name",
                "email",
                "subject",
                "body"
            ]
        )

    if data["flagged_leads"]:

        flagged_df = pd.DataFrame(
            data["flagged_leads"]
        )[
            [
                "contact_name",
                "email",
                "revisions"
            ]
        ]

    else:

        flagged_df = pd.DataFrame(
            columns=[
                "contact_name",
                "email",
                "revisions"
            ]
        )

    return (
        summary,
        outbox_df,
        flagged_df
    )


# ---------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------

with gr.Blocks(
    title="Sales Outreach Agent"
) as demo:

    gr.Markdown(
        """
        ## Sales Outreach Agent

        Upload your leads CSV and generate
        personalized AI-powered outreach emails.
        """
    )

    with gr.Row():

        company_input = gr.Textbox(
            label="Your company name",
            placeholder="e.g. upGrad"
        )

        role_input = gr.Textbox(
            label="Your role",
            placeholder="e.g. Admissions Counsellor"
        )

    csv_input = gr.File(
        label="Leads CSV",
        file_types=[".csv"],
        type="filepath"
    )

    run_btn = gr.Button(
        "Generate emails",
        variant="primary"
    )

    summary_out = gr.Textbox(
        label="Summary",
        interactive=False
    )

    gr.Markdown(
        "### Approved (outbox)"
    )

    outbox_out = gr.Dataframe(
        wrap=True
    )

    gr.Markdown(
        "### Flagged (needs manual review)"
    )

    flagged_out = gr.Dataframe(
        wrap=True
    )

    run_btn.click(
        fn=run_agent,
        inputs=[
            csv_input,
            company_input,
            role_input
        ],
        outputs=[
            summary_out,
            outbox_out,
            flagged_out
        ]
    )


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )