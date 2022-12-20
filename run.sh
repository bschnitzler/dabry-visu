PYTHONPATH=$(cat MERMOZ_PATH)/src:$(pwd)/src MERMOZ_PATH=$(cat MERMOZ_PATH) python main.py "$@"
