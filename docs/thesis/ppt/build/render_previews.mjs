import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname));
const PPT_DIR = path.resolve(BUILD_DIR, "..");
const INPUT_PPTX = path.join(PPT_DIR, process.argv[2] || "proposal-defense-classic-clean.pptx");
const PREVIEW_DIR = path.join(PPT_DIR, "preview");

function fileUrlPath(rawPath) {
  if (process.platform === "win32" && rawPath.startsWith("/")) {
    return rawPath.slice(1);
  }
  return rawPath;
}

async function saveExportedBlob(blob, outputPath) {
  if (typeof blob.save === "function") {
    await blob.save(outputPath);
    return;
  }
  const arrayBuffer = await blob.arrayBuffer();
  await fs.writeFile(outputPath, Buffer.from(arrayBuffer));
}

async function main() {
  const inputPath = fileUrlPath(INPUT_PPTX);
  const previewDir = fileUrlPath(PREVIEW_DIR);
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPath));
  const outputs = [];
  for (let index = 0; index < presentation.slides.items.length; index += 1) {
    const slide = presentation.slides.items[index];
    const blob = await presentation.export({ slide, format: "png", scale: 1 });
    const outputPath = path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`);
    await saveExportedBlob(blob, outputPath);
    outputs.push(outputPath);
  }
  console.log(outputs.join("\n"));
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
