# Makefile

POSTS_DIR := _posts
IMAGES_DIR := images
DOWNLOADS_DIR := $(HOME)/Downloads

# Get the latest .ipynb notebook from Downloads
LATEST_NOTEBOOK := $(shell ls -t $(DOWNLOADS_DIR)/*.ipynb 2>/dev/null | head -n 1)
NOTEBOOK_FILENAME := $(notdir $(LATEST_NOTEBOOK))

# Convert underscores to dashes for Markdown filename
DASHED_NAME := $(shell echo $(NOTEBOOK_FILENAME:.ipynb=) | tr '_' '-')
MARKDOWN_FILENAME := $(DASHED_NAME).md
MARKDOWN_PATH := $(POSTS_DIR)/$(MARKDOWN_FILENAME)

# Extract date for image folder (first three underscore-separated parts)
DATE := $(shell echo $(NOTEBOOK_FILENAME) | cut -d'_' -f1,2,3 | tr '_' '-')
IMAGE_OUTPUT_DIR := $(IMAGES_DIR)/$(DATE)

# 🆘 Default target: show help
.DEFAULT_GOAL := help

.PHONY: help post format all 

help:
	@echo "Available targets:"
	@echo "  post             Download latest .ipynb from ~/Downloads, convert to markdown, and move images."
	@echo "  format           Add YAML front matter and format image links in latest converted post."
	@echo "  all  Run both post and format steps in sequence."

post:
	@if [ -z "$(LATEST_NOTEBOOK)" ]; then \
		echo "❌ No notebook found in $(DOWNLOADS_DIR)"; \
		exit 1; \
	fi

	@echo "📥 Copying notebook: $(NOTEBOOK_FILENAME)"
	cp "$(LATEST_NOTEBOOK)" .

	@echo "🔄 Converting notebook to markdown..."
	jupyter nbconvert --to markdown "$(NOTEBOOK_FILENAME)" \
		--output="$(DASHED_NAME)" --output-dir="$(POSTS_DIR)" \
		--ExtractOutputPreprocessor.enabled=True

	@echo "📁 Creating image output directory: $(IMAGE_OUTPUT_DIR)"
	mkdir -p "$(IMAGE_OUTPUT_DIR)"

	@echo "📂 Moving extracted images..."
	@if ls "$(POSTS_DIR)"/*_files >/dev/null 2>&1; then \
		mv "$(POSTS_DIR)"/*_files/* "$(IMAGE_OUTPUT_DIR)/"; \
		rm -r "$(POSTS_DIR)"/*_files; \
	else \
		echo "ℹ️  No images to move."; \
	fi

	@echo "✅ Done: Markdown post saved to $(MARKDOWN_PATH)"

format:
	@echo "📝 Formatting markdown..."

	@if [ -z "$(MARKDOWN_PATH)" ]; then \
		echo "❌ Markdown file not found."; \
		exit 1; \
	fi

	@echo "📋 Adding front matter to: $(MARKDOWN_PATH)"
	@DATE_ONLY=$(DATE); \
		TITLE=$$(echo "$(DASHED_NAME)" | cut -d'-' -f4- | sed 's/-/ /g'); \
		FM="---\nlayout: post\ntitle: \"$$TITLE\"\ndate: $$DATE_ONLY\ncategories: [R, Python]\ncomments: true\n---\n\n"; \
		TMPFILE=$$(mktemp); \
		echo "$$FM" > "$$TMPFILE"; \
		cat "$(MARKDOWN_PATH)" >> "$$TMPFILE"; \
		mv "$$TMPFILE" "$(MARKDOWN_PATH)"

	@echo "🔄 Rewriting image links..."
	@perl -pi -e 's|!\[.*?\]\((.*?_files/.*?\.png)\)|"![" . "image-title-here]({{base}}/images/$(DATE)/" . $$1 =~ s!.*/!!r . "){:class=\"img-responsive\"}"|eg' "$(MARKDOWN_PATH)"

	@echo "✅ Formatting complete."

all: post format
	@echo "🧹 Cleaning up original notebook..."
	@rm -f "$(NOTEBOOK_FILENAME)"
	@echo "✅ All done."
