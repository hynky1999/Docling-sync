sudo apt-get update
sudo apt-get install -y  python3.12-venv

python -m venv venv
source venv/bin/activate
pip install git+https://github.com/huggingface/datatrove.git@pdfs-branch


# Docling stuff
git clone https://github.com/hynky1999/Docling-sync
pip install -e Docling-sync/docling/
pip install -e Docling-sync/docling-ibm-models/
pip install openvino zstandard warcio s3fs pymupdf orjson

export LAYOUT_VINO_PATH="./Docling-sync/models/v2-quant.xml"
python print_info.py
python run_extract.py