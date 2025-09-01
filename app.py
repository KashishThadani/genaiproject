# app.py
# AI-driven Game Character Design Generator (Streamlit)
#
# Features:
# - Procedural character generator (genre, species/class, traits, stats, backstory)
# - Prompt builder for image tools (Stable Diffusion / Midjourney / DALL·E, etc.)
# - Optional local image generation via HuggingFace diffusers (Stable Diffusion)
# - Export: JSON, prompt, and PNG concept card
# - Designed to run locally or on Streamlit Community Cloud (image gen can be disabled)
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Streamlit Cloud deploy:
#   1) Push app.py and requirements.txt to a Git repo
#   2) Deploy on share.streamlit.io
#   3) In "Settings → Secrets", add HF_TOKEN (optional) if you switch to an Inference API
#      or private model. For local diffusers, no token is required for public models.
#
# Notes:
# - Image generation is optional and off by default. Turn it on via the sidebar.
# - If diffusers/torch aren't available (e.g., Streamlit Cloud), the app gracefully skips images.
# - You can still export prompts and JSON, and use external tools for art.
#
# (c) 2025 – MIT License

import io
import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ----------------------------- Utilities -----------------------------

APP_TITLE = "AI Game Character Designer"
VERSION = "1.0.0"

def seeded_rng(seed: Optional[int] = None) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(seed)
    return r

def pick(r: random.Random, items: List[str]) -> str:
    return r.choice(items)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ----------------------------- Data -----------------------------

GENRES = [
    "Fantasy", "Sci‑Fi", "Post‑Apocalyptic", "Cyberpunk",
    "Steampunk", "Historical", "Horror", "Modern", "Mythic"
]

ART_STYLES = [
    "Realistic", "Cel‑shaded", "Pixel Art", "Anime", "Comic/Ink",
    "Painterly", "Low‑poly 3D", "Stylized 3D", "Watercolor"
]

SPECIES_BY_GENRE = {
    "Fantasy": ["Human", "Elf", "Dwarf", "Orc", "Tiefling", "Dragonborn", "Halfling"],
    "Sci‑Fi": ["Human", "Android", "Cyborg", "Alien (humanoid)", "Alien (xeno)", "Clone"],
    "Post‑Apocalyptic": ["Human", "Mutant", "Android", "Cyborg", "Ghoul"],
    "Cyberpunk": ["Human", "Augmented Human", "Cyborg", "Android"],
    "Steampunk": ["Human", "Automaton", "Aether‑touched"],
    "Historical": ["Human"],
    "Horror": ["Human", "Vampire", "Werewolf", "Ghost", "Demon‑kin"],
    "Modern": ["Human"],
    "Mythic": ["Human", "Demigod", "Fae", "Satyr", "Nymph", "Gorgon"]
}

CLASSES_BY_GENRE = {
    "Fantasy": ["Warrior", "Ranger", "Rogue", "Mage", "Cleric", "Bard", "Monk"],
    "Sci‑Fi": ["Pilot", "Engineer", "Hacker", "Soldier", "Scientist", "Medic", "Psionic"],
    "Post‑Apocalyptic": ["Scavenger", "Raider", "Warden", "Mechanic", "Medic", "Stalker"],
    "Cyberpunk": ["Netrunner", "Solo", "Techie", "Fixer", "Media", "Medtech"],
    "Steampunk": ["Inventor", "Airship Captain", "Alchemist", "Mechanist", "Explorer"],
    "Historical": ["Knight", "Archer", "Scholar", "Merchant", "Assassin"],
    "Horror": ["Occultist", "Hunter", "Medium", "Exorcist", "Cultist"],
    "Modern": ["Detective", "Soldier", "Athlete", "Artist", "Scientist", "Doctor"],
    "Mythic": ["Champion", "Oracle", "Trickster", "Guardian", "Warden"]
}

ALIGNMENTS = [
    "Lawful Good", "Neutral Good", "Chaotic Good",
    "Lawful Neutral", "True Neutral", "Chaotic Neutral",
    "Lawful Evil", "Neutral Evil", "Chaotic Evil"
]

TRAITS_POS = [
    "brave", "curious", "loyal", "resilient", "clever", "empathetic",
    "disciplined", "charismatic", "resourceful", "perceptive"
]
TRAITS_NEG = [
    "impulsive", "stubborn", "secretive", "reckless", "cynical", "arrogant",
    "overprotective", "superstitious", "vengeful", "gullible"
]

QUIRKS = [
    "collects shiny trinkets", "hums battle hymns", "talks to machines",
    "keeps a coded diary", "never removes gloves", "feeds stray animals",
    "quotes obscure proverbs", "counts steps before acting", "sketches allies",
    "brews custom tea blends"
]

COLORS = [
    "crimson", "emerald", "sapphire", "amber", "violet", "teal",
    "ivory", "obsidian", "cobalt", "brass", "bronze", "silver"
]

WEAPONS = [
    "longsword", "greatsword", "daggers", "bow", "crossbow", "spear",
    "staff", "handgun", "rifle", "plasma blade", "railgun", "shock baton"
]

GEAR = [
    "cloak", "hood", "goggles", "utility belt", "pauldrons", "greaves",
    "holo‑wrist", "medkit", "datapad", "gas mask", "backpack", "gloves"
]

GOALS = [
    "avenge a fallen mentor", "restore a lost relic", "protect their home",
    "decode a forbidden archive", "map the uncharted", "cure a spreading plague",
    "break a family curse", "dismantle a corrupted guild", "win the Tournament",
    "earn redemption for past crimes"
]

BACKSTORY_HOOKS = [
    "was the sole survivor of a siege",
    "grew up in a floating city of traders",
    "escaped a clandestine lab with altered abilities",
    "made a pact with an ancient entity",
    "betrayed their order to save civilians",
    "was raised by nomads who navigate the storms",
    "inherited a map etched into their skin",
    "can hear whispers from relics of the old world"
]


# ----------------------------- Models -----------------------------

@dataclass
class Stats:
    strength: int
    agility: int
    intellect: int
    charisma: int
    endurance: int
    luck: int

@dataclass
class Character:
    name: str
    genre: str
    art_style: str
    species: str
    char_class: str
    alignment: str
    age: int
    height_cm: int
    build: str
    eye_color: str
    hair_color: str
    primary_colors: List[str]
    traits_positive: List[str]
    traits_negative: List[str]
    quirks: List[str]
    signature_weapon: str
    notable_gear: List[str]
    goal: str
    backstory: str
    stats: Stats
    prompt: str


# ----------------------------- Generators -----------------------------

SYLLABLES = ["ka","shi","ra","den","mar","vel","tor","zan","rei","lyn","vor","tia","sel","ith","nar","dra","quo","xin","bal","mia"]

def gen_name(r: random.Random):
    parts = r.randint(2, 3)
    name = "".join(r.choice(SYLLABLES) for _ in range(parts)).capitalize()
    # Optionally add a surname
    if r.random() < 0.5:
        surname = "".join(r.choice(SYLLABLES) for _ in range(parts)).capitalize()
        return f"{name} {surname}"
    return name

BUILDS = ["slender", "athletic", "stocky", "towering", "lithe", "bulky"]
EYE_COLORS = ["amber", "emerald", "ice‑blue", "violet", "hazel", "obsidian"]
HAIR_COLORS = ["silver", "jet‑black", "auburn", "white", "ash‑blonde", "teal‑dyed"]

def gen_palette(r: random.Random, n=3):
    # pick 3 distinct color words
    return r.sample(COLORS, k=min(n, len(COLORS)))

def stat_block(r: random.Random, base: int, spread: int) -> Stats:
    return Stats(
        strength=clamp(int(r.gauss(base, spread)), 1, 20),
        agility=clamp(int(r.gauss(base, spread)), 1, 20),
        intellect=clamp(int(r.gauss(base, spread)), 1, 20),
        charisma=clamp(int(r.gauss(base, spread)), 1, 20),
        endurance=clamp(int(r.gauss(base, spread)), 1, 20),
        luck=clamp(int(r.gauss(base, spread)), 1, 20),
    )

def build_prompt(c: Character) -> str:
    # Highly structured prompt suitable for SD/MJ/DALL·E
    style_hint = f"in {c.art_style} style"
    palette = ", ".join(c.primary_colors)
    gear = ", ".join(c.notable_gear)
    traits = ", ".join(c.traits_positive + c.traits_negative)
    return (
        f"{c.genre} game character portrait, {c.species} {c.char_class}, "
        f"alignment {c.alignment}, age {c.age}, {c.build} build, height {c.height_cm} cm, "
        f"{c.hair_color} hair, {c.eye_color} eyes, palette: {palette}. "
        f"Wielding {c.signature_weapon}, wearing {gear}. Personality: {traits}. "
        f"Intriguing backstory: {c.backstory}. Highly detailed, {style_hint}, character turnaround sheet."
    )

def generate_character(
    seed: Optional[int],
    genre: str,
    style: str,
    custom_species: Optional[str] = None,
    custom_class: Optional[str] = None,
    align: Optional[str] = None,
    stat_bias: str = "Balanced",
) -> Character:
    r = seeded_rng(seed)
    species_list = SPECIES_BY_GENRE.get(genre, ["Human"])
    _species = custom_species or pick(r, species_list)
    class_list = CLASSES_BY_GENRE.get(genre, ["Adventurer"])
    _class = custom_class or pick(r, class_list)
    _align = align or pick(r, ALIGNMENTS)

    # Stat biases
    base, spread = 10, 3
    if stat_bias == "Strength":
        base, spread = 12, 3
    elif stat_bias == "Agility":
        base, spread = 12, 3
    elif stat_bias == "Intellect":
        base, spread = 12, 3
    elif stat_bias == "Charisma":
        base, spread = 12, 3
    elif stat_bias == "Endurance":
        base, spread = 12, 3
    elif stat_bias == "Luck":
        base, spread = 12, 3

    name = gen_name(r)
    height = r.randint(140, 210)
    age = r.randint(16, 80)
    build = pick(r, BUILDS)
    eyes = pick(r, EYE_COLORS)
    hair = pick(r, HAIR_COLORS)
    palette = gen_palette(r, 3)

    traits_pos = r.sample(TRAITS_POS, k=3)
    traits_neg = r.sample(TRAITS_NEG, k=2)
    quirks = r.sample(QUIRKS, k=2)
    weapon = pick(r, WEAPONS)
    gear = r.sample(GEAR, k=3)
    goal = pick(r, GOALS)
    hook = pick(r, BACKSTORY_HOOKS)

    backstory = (
        f"{name} {_species} {_class} {_align.lower()} who {hook}. "
        f"Their goal is to {goal}. They are {', '.join(traits_pos)} yet "
        f"{', '.join(traits_neg)}; they {quirks[0]} and {quirks[1]}. "
        f"They favor {weapon} and carry {', '.join(gear)}."
    )

    stats = stat_block(r, base, spread)

    char = Character(
        name=name,
        genre=genre,
        art_style=style,
        species=_species,
        char_class=_class,
        alignment=_align,
        age=age,
        height_cm=height,
        build=build,
        eye_color=eyes,
        hair_color=hair,
        primary_colors=palette,
        traits_positive=traits_pos,
        traits_negative=traits_neg,
        quirks=quirks,
        signature_weapon=weapon,
        notable_gear=gear,
        goal=goal,
        backstory=backstory,
        stats=stats,
        prompt="",  # filled below
    )
    char.prompt = build_prompt(char)
    return char


# ----------------------------- Optional Image Gen (diffusers) -----------------------------

def try_import_diffusers():
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        return StableDiffusionPipeline, torch
    except Exception:
        return None, None

def generate_image_with_diffusers(prompt: str, seed: Optional[int], steps: int = 25, guidance: float = 7.0, width: int = 768, height: int = 768):
    SD, torch = try_import_diffusers()
    if SD is None:
        return None, "diffusers/torch not installed; install requirements and ensure a GPU for best results."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = SD.from_pretrained("runwayml/stable-diffusion-v1-5")
        if device == "cuda":
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        generator = torch.Generator(device=device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        return image, None
    except Exception as e:
        return None, f"Image generation failed: {e}"


# ----------------------------- Concept Card -----------------------------

def render_concept_card(c: Character, img: Optional[Image.Image] = None) -> Image.Image:
    W, H = (1024, 768)
    card = Image.new("RGB", (W, H), (18, 18, 20))
    draw = ImageDraw.Draw(card)

    # Try a default PIL font
    try:
        title_font = ImageFont.truetype("arial.ttf", 36)
        body_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    padding = 24
    x = padding
    y = padding

    # Title
    draw.text((x, y), f"{c.name} — {c.species} {c.char_class}", fill=(240, 240, 240), font=title_font)
    y += 48

    # Left column text
    left_text = (
        f"Genre: {c.genre}\n"
        f"Style: {c.art_style}\n"
        f"Alignment: {c.alignment}\n"
        f"Age/Height/Build: {c.age}, {c.height_cm} cm, {c.build}\n"
        f"Eyes/Hair: {c.eye_color}, {c.hair_color}\n"
        f"Palette: {', '.join(c.primary_colors)}\n"
        f"Weapon: {c.signature_weapon}\n"
        f"Gear: {', '.join(c.notable_gear)}\n"
        f"Traits+: {', '.join(c.traits_positive)}\n"
        f"Traits‑: {', '.join(c.traits_negative)}\n"
        f"Quirks: {', '.join(c.quirks)}\n"
        f"Goal: {c.goal}\n"
    )
    draw.multiline_text((x, y), left_text, fill=(220, 220, 220), font=body_font, spacing=4)

    # Right column: image or backstory box
    rx = W // 2
    ry = padding + 48
    box_w = W - rx - padding
    box_h = H - ry - padding

    if img is not None:
        img_aspect = img.width / img.height
        target_w = box_w
        target_h = int(target_w / img_aspect)
        if target_h > box_h:
            target_h = box_h
            target_w = int(target_h * img_aspect)
        img_resized = img.resize((target_w, target_h))
        card.paste(img_resized, (rx + (box_w - target_w)//2, ry + (box_h - target_h)//2))
    else:
        # Backstory text block
        story = f"Backstory:\n{c.backstory}"
        draw.multiline_text((rx, ry), story, fill=(210, 210, 210), font=body_font, spacing=6)

    # Footer
    footer = f"{APP_TITLE} v{VERSION} — Generated {datetime.date.today().isoformat()}"
    draw.text((x, H - padding - 16), footer, fill=(160, 160, 160), font=body_font)

    return card


# ----------------------------- Streamlit UI -----------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🎮", layout="wide")

with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Everything is optional — use Randomize or tweak as you like.")
    seed = st.number_input("Random seed (optional)", min_value=0, max_value=2_000_000_000, value=0, step=1)
    use_seed = st.toggle("Use seed", value=False)
    chosen_seed = int(seed) if use_seed else None

    genre = st.selectbox("Genre", GENRES, index=0)
    style = st.selectbox("Art Style", ART_STYLES, index=0)

    # Species/Class – can pick from curated list or type custom
    species_list = SPECIES_BY_GENRE.get(genre, ["Human"])
    species_choice = st.selectbox("Species (preset)", species_list)
    species_custom = st.text_input("...or custom species", "")

    class_list = CLASSES_BY_GENRE.get(genre, ["Adventurer"])
    class_choice = st.selectbox("Class/Role (preset)", class_list)
    class_custom = st.text_input("...or custom class/role", "")

    align = st.selectbox("Alignment", ALIGNMENTS, index=3)
    stat_bias = st.selectbox("Primary Stat Bias", ["Balanced","Strength","Agility","Intellect","Charisma","Endurance","Luck"], index=0)

    enable_images = st.toggle("Enable local image generation (diffusers)", value=False)
    steps = st.slider("Diffusion steps", 10, 60, 25) if enable_images else 25
    guidance = st.slider("Guidance scale", 1.0, 12.0, 7.0) if enable_images else 7.0
    width = st.select_slider("Image width", [512, 640, 768, 896], value=768) if enable_images else 768
    height = st.select_slider("Image height", [512, 640, 768, 896], value=768) if enable_images else 768

st.title("🎮 AI Game Character Designer")
st.write("Create rich, game‑ready character sheets with prompts and optional concept art.")

if "char" not in st.session_state:
    st.session_state.char = None
if "img" not in st.session_state:
    st.session_state.img = None

col1, col2 = st.columns([1,1])

with col1:
    if st.button("🎲 Generate Character"):
        c = generate_character(
            seed=chosen_seed,
            genre=genre,
            style=style,
            custom_species=species_custom or None,
            custom_class=class_custom or None,
            align=align,
            stat_bias=stat_bias,
        )
        st.session_state.char = c
        st.session_state.img = None

    if st.session_state.char is not None:
        c = st.session_state.char
        st.subheader(f"{c.name} — {c.species} {c.char_class}")
        st.text_area("Backstory", c.backstory, height=200)
        st.write("**Core Stats**")
        s = c.stats
        st.write(
            f"STR {s.strength} | AGI {s.agility} | INT {s.intellect} | CHA {s.charisma} | END {s.endurance} | LUCK {s.luck}"
        )
        st.write("**Appearance & Gear**")
        st.write(
            f"Age {c.age}, Height {c.height_cm} cm, Build {c.build}, Eyes {c.eye_color}, Hair {c.hair_color}"
        )
        st.write(f"Palette: {', '.join(c.primary_colors)}")
        st.write(f"Weapon: {c.signature_weapon} | Gear: {', '.join(c.notable_gear)}")
        st.write(f"Traits: {', '.join(c.traits_positive)} | Flaws: {', '.join(c.traits_negative)} | Quirks: {', '.join(c.quirks)}")

with col2:
    if st.session_state.char is not None:
        c = st.session_state.char
        st.subheader("🎨 Image / Prompt")

        st.text_area("🔧 Prompt (copy to Midjourney/SD/DALL·E)", c.prompt, height=180)

        gen_img = st.button("🖼️ Generate Concept Image (local diffusers)", disabled=not enable_images)
        if gen_img and enable_images:
            with st.spinner("Running Stable Diffusion..."):
                img, err = generate_image_with_diffusers(c.prompt, chosen_seed, steps, guidance, width, height)
                if err:
                    st.warning(err)
                else:
                    st.session_state.img = img

        if st.session_state.img is not None:
            st.image(st.session_state.img, caption="Concept Image")

        # Concept card render & downloads
        card = render_concept_card(c, st.session_state.img)
        buf = io.BytesIO()
        card.save(buf, format="PNG")
        card_bytes = buf.getvalue()

        st.download_button("⬇️ Download Concept Card (PNG)", data=card_bytes, file_name=f"{c.name.replace(' ','_')}_concept.png", mime="image/png")

        # JSON export
        json_bytes = json.dumps(asdict(c), indent=2).encode("utf-8")
        st.download_button("⬇️ Download Character JSON", data=json_bytes, file_name=f"{c.name.replace(' ','_')}.json", mime="application/json")

        # Prompt export
        st.download_button("⬇️ Download Prompt (.txt)", data=c.prompt.encode("utf-8"), file_name=f"{c.name.replace(' ','_')}_prompt.txt", mime="text/plain")

st.divider()
with st.expander("Tips & Notes"):
    st.markdown("""
- **Reproducibility**: toggle *Use seed* in the sidebar; same seed ➜ same character.
- **Image generation** is optional. For the cloud, you can keep it off and still export prompts/JSON.
- To speed up local images, prefer a **GPU** (CUDA). CPU will be slow.
- You can change models in the code (`runwayml/stable-diffusion-v1-5`) to other checkpoints.
""")
