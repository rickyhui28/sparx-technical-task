ENV_NAME=env
WORKING_DIR=working
NOTEBOOK=$(WORKING_DIR)/analysis.ipynb
EXPORT_DIR=exports

# Create virtual environment and install dependencies
install:
	python3 -m venv $(ENV_NAME)
	. $(ENV_NAME)/bin/activate && pip3 install -r requirements.txt

# Launch the main analysis notebook
notebook:
	. $(ENV_NAME)/bin/activate && jupyter notebook $(NOTEBOOK)

# Export notebook to clean HTML (markdown + outputs, no code)
export-summary:
	mkdir -p $(EXPORT_DIR)
	jupyter nbconvert --to html --output $(EXPORT_DIR)/summary.html \
		--TemplateExporter.exclude_input=True $(NOTEBOOK)

# Clean up the environment and cache files
clean:
	rm -rf $(ENV_NAME)
	find . -type d -name "__pycache__" -exec rm -r {} +


