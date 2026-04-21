import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname));
const PPT_DIR = path.resolve(BUILD_DIR, "..");
const REPO_ROOT = path.resolve(PPT_DIR, "..", "..", "..");
const INPUT_PPTX = path.join(PPT_DIR, "proposal-defense-classic-clean.pptx");
const OUTPUT_PPTX = INPUT_PPTX;

function fileUrlPath(rawPath) {
  if (process.platform === "win32" && rawPath.startsWith("/")) {
    return rawPath.slice(1);
  }
  return rawPath;
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function imageConfig(imagePath) {
  return {
    blob: readImageBlob(imagePath),
    fit: "contain",
    alt: path.basename(imagePath),
  };
}

function addImage(slide, imagePath, left, top, width, height) {
  const image = slide.images.add({
    blob: undefined,
    fit: "contain",
    alt: path.basename(imagePath),
  });
  return { image, imagePath, position: { left, top, width, height } };
}

async function placeImage(slide, imagePath, left, top, width, height) {
  const image = slide.images.add({
    blob: await readImageBlob(imagePath),
    fit: "contain",
    alt: path.basename(imagePath),
  });
  image.position = { left, top, width, height };
}

async function main() {
  const inputPath = fileUrlPath(INPUT_PPTX);
  const outputPath = fileUrlPath(OUTPUT_PPTX);
  const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPath));
  const slideSize = presentation.slideSize || presentation.options?.slideSize || null;
  if (slideSize) {
    console.log(`slideSize=${slideSize.width}x${slideSize.height}`);
  }

  const glyphDir = path.join(REPO_ROOT, "data", "generated", "sample_dataset", "images", "Hiragino_Sans_GB");
  const componentDir = path.join(REPO_ROOT, "data", "generated", "sample_dataset", "components", "Hiragino_Sans_GB");
  const slides = presentation.slides.items;

  const cover = slides[0];
  await placeImage(cover, path.join(glyphDir, "狗_glyph.png"), 1339, 334, 122, 118);
  await placeImage(cover, path.join(glyphDir, "困_glyph.png"), 1504, 334, 122, 118);
  await placeImage(cover, path.join(glyphDir, "间_glyph.png"), 1339, 554, 122, 118);
  await placeImage(cover, path.join(glyphDir, "明_glyph.png"), 1504, 554, 122, 118);

  const background = slides[2];
  await placeImage(background, path.join(glyphDir, "狗_glyph.png"), 1159, 344, 202, 154);
  await placeImage(background, path.join(glyphDir, "困_glyph.png"), 1459, 344, 202, 154);
  await placeImage(background, path.join(glyphDir, "间_glyph.png"), 1159, 624, 202, 154);
  await placeImage(background, path.join(glyphDir, "闪_glyph.png"), 1459, 624, 202, 154);

  const task = slides[5];
  await placeImage(task, path.join(componentDir, "狗_a.png"), 164, 329, 132, 112);
  await placeImage(task, path.join(componentDir, "狗_b.png"), 429, 329, 132, 112);
  await placeImage(task, path.join(glyphDir, "狗_glyph.png"), 1024, 314, 162, 154);
  await placeImage(task, path.join(componentDir, "困_a.png"), 164, 649, 132, 112);
  await placeImage(task, path.join(componentDir, "困_b.png"), 429, 649, 132, 112);
  await placeImage(task, path.join(glyphDir, "困_glyph.png"), 1044, 634, 162, 154);

  const exported = await PresentationFile.exportPptx(presentation);
  await exported.save(outputPath);
  console.log(`Updated clean deck with images at ${outputPath}`);
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
