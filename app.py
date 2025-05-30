import os
import random
import requests
import gradio as gr
from PIL import Image
from io import BytesIO
from gradio_client import Client


sponsor_html = """
<div style="display:flex; padding: 0em; justify-content: center; gap: 1em; border-radius: 2em;">
  <img src="https://static-00.iconduck.com/assets.00/google-cloud-icon-2048x1288-h9qynww8.png"
       style="height:1em; width:auto; object-fit:contain;"
       title="Google Cloud for Startups"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/2560px-Amazon_Web_Services_Logo.svg.png"
       style="height:1em; width:auto; object-fit:contain;"
       title="AWS Activate"/>
  <img src="https://ageyetech.com/wp-content/uploads/2020/07/AgEye_nvidia_inception_logo_new.png"
       style="height:1em; width:auto; object-fit:contain;"
       title="NVIDIA Inception"/>
  <img src="https://azurecomcdn.azureedge.net/cvt-8310f955fa0c7812bd316a20d46a917e5b94170e9e9da481ca3045acae446bb5/svg/logo.svg"
       style="height:1em; width:auto; object-fit:contain;"
       title="Azure for Startups"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Cloudflare_Logo.svg/2560px-Cloudflare_Logo.svg.png"
       style="height:1em; width=auto; object-fit:contain;"
       title="Cloudflare"/>
  <img src="https://scaleway.com/cdn-cgi/image/width=640/https://www-uploads.scaleway.com/Scaleway_3_D_Logo_57e7fb833f.png"
       style="height:1em; width:auto; object-fit:contain;"
       title="Scaleway"/>
  <img src="https://cdn.prod.website-files.com/63e26df0d6659968e46142f7/63e27b40e661321d5278519b_logotype-bb8cd083.svg"
       style="height:1em; width:auto; object-fit:contain;"
       title="Modal"/>
  <img src="https://pollinations.ai/favicon.ico"
       style="height:1em; width:auto; object-fit:contain;"
       title="Pollination.ai"/>
</div>
"""

# more servers coming soon...


SERVER_NAMES = {
    "google_us": "Google US Server",
    "azure_lite": "Azure Lite Supercomputer Server",
    "artemis" : "Artemis GPU Super cluster",
    "nb_dr" : "NebulaDrive Tensor Server",
    "pixelnet" : "PixelNet NPU Server",
    "nsfw_core" : "NSFW-Core: Uncensored Server",
    "nsfw_core_2" : "NSFW-Core: Uncensored Server 2",
    "nsfw_core_3" : "NSFW-Core: Uncensored Server 3",
    "nsfw_core_4" : "NSFW-Core: Uncensored Server 4",
}


SERVER_SOCKETS = {
    "google_us": None,
    "azure_lite": "FLUX-Pro-SERVER1",
    "artemis" : "FLUX-Pro-Artemis-GPU",
    "nb_dr" : "FLUX-Pro-NEBULADRIVE",
    "pixelnet" : "FLUX-Pro-PIXELNET",
    "nsfw_core": "FLUX-Pro-NSFW-LocalCoreProcessor",
    "nsfw_core_2" : "FLUX-Pro-NSFW-LocalCoreProcessor-v2",
    "nsfw_core_3" : "FLUX-Pro-NSFW-LocalCoreProcessor-v3",
    "nsfw_core_4" : "FLUX-Pro-NSFW-LocalCoreProcessor-v4",
}

HF_TOKEN = os.environ.get("HF_TOKEN")
FLUX_URL  = os.environ.get("FLUX_URL")


def _open_image_from_str(s: str):
    # base64 decoding
    if s.startswith("http"):
        r = requests.get(s); return Image.open(BytesIO(r.content))
    if os.path.exists(s):
        return Image.open(s)
    # try base64 blob
    try:
        import base64
        _, b64 = s.split(",", 1)
        data = base64.b64decode(b64)
        return Image.open(BytesIO(data))
    except:
        raise ValueError(f"Can't parse image string: {s[:30]}…")


def generate_image(prompt, width, height, seed, randomize, server_choice):

    print(prompt+"\n\n\n\n")
    # determine seed
    if randomize:
        seed = random.randint(0, 9_999_999)
    used_seed = seed

    # pick server key and socket
    key = next(k for k, v in SERVER_NAMES.items() if v == server_choice)
    socket = SERVER_SOCKETS.get(key)

    # generate image via FLUX or HF space
    if socket is None:
        if not FLUX_URL:
            return "Error: FLUX_URL not set.", used_seed
        url = (
            FLUX_URL
            .replace("[prompt]", prompt)
            .replace("[w]", str(width))
            .replace("[h]", str(height))
            .replace("[seed]", str(seed))
        )
        r = requests.get(url)
        img = Image.open(BytesIO(r.content)) if r.ok else f"FLUX-Pro failed ({r.status_code})"
    else:
        space_id = f"NihalGazi/{socket}"
        client = Client(space_id, hf_token=HF_TOKEN)
        res = client.predict(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
            randomize=randomize,
            api_name="/predict"
        )
        if isinstance(res, dict):
            if res.get("path"):
                img = Image.open(res["path"])
            elif res.get("url"):
                img = _open_image_from_str(res["url"])
            else:
                img = "No image found in response."
        elif isinstance(res, str):
            img = _open_image_from_str(res)
        else:
            img = f"Unexpected response type: {type(res)}"

    # return both image and used seed
    return img, used_seed

# ─── GRADIO INTERFACE ─────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown(
        """
# Unlimited FLUX-Pro

**Enter a prompt and tweak your settings:**  
- **Width & Height** – choose your canvas size  
- **Seed** – pick a number or check **Randomize Seed**  
- **Server** – switch between servers if one is slow or fails:
  - **Google US Server**  
  - **Azure Lite Supercomputer Server**  
  - **Artemis GPU Super cluster**  
  - **NebulaDrive Tensor Server**  
  - **PixelNet NPU Server**  
  - **NSFW‑Core: Uncensored Servers** (for explicit content; use responsibly)
- **Suggestions** – have ideas? I'm open to them!

⚠️ **Caution:**  
The **NSFW‑Core** server can generate adult‑only content. You must be of legal age in your jurisdiction and comply with all local laws and platform policies. Developer is not liable for misuse.


> ⚡ 4 NSFW Servers available 


Click **Generate** and enjoy unlimited AI art!

❤️ **Like & follow** for more AI projects:  
• Instagram: [@nihal_gazi_io](https://www.instagram.com/nihal_gazi_io/)  
• Discord: nihal_gazi_io  
• Enjoying this? [Support](https://huggingface.co/spaces/NihalGazi/FLUX-Pro-Unlimited/discussions/11#68319a4ffcf0624d8b93d424) the creator with a small donation — every bit helps!


"""
    )

    # Inputs
    prompt = gr.Textbox(label="Prompt", placeholder="Enter your image prompt…", lines=4)
    width  = gr.Slider(512, 2048, step=16, value=1280, label="Width")
    height = gr.Slider(512, 2048, step=16, value=1280, label="Height")
    seed   = gr.Number(label="Seed", value=0)
    rand   = gr.Checkbox(label="Randomize Seed", value=True)
    server = gr.Dropdown(label="Server", choices=list(SERVER_NAMES.values()),
                         value=list(SERVER_NAMES.values())[0])

    generate_btn = gr.Button("Generate")

    # Outputs: image and seed display
    output = gr.Image(type="pil", label="Generated Image")
    seed_display = gr.Textbox(label="Used Seed", interactive=False)

    generate_btn.click(
        generate_image,
        inputs=[prompt, width, height, seed, rand, server],
        outputs=[output, seed_display],
        concurrency_limit=None
    )

    # Sponsor wall
    gr.HTML(sponsor_html)

demo.launch() 