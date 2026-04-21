import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const BUILD_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname));
const PPT_DIR = path.resolve(BUILD_DIR, "..");
const INPUT_PPTX = path.join(PPT_DIR, "proposal-defense-classic-clean.pptx");
const OUTPUT_PPTX = path.join(PPT_DIR, "proposal-defense-classic-notes.pptx");
const NOTES_PATH = path.join(PPT_DIR, "speaker_notes.json");

function fileUrlPath(rawPath) {
  if (process.platform === "win32" && rawPath.startsWith("/")) {
    return rawPath.slice(1);
  }
  return rawPath;
}

async function main() {
  const inputPath = fileUrlPath(INPUT_PPTX);
  const outputPath = fileUrlPath(OUTPUT_PPTX);
  const notesPath = fileUrlPath(NOTES_PATH);

  const notes = JSON.parse(await fs.readFile(notesPath, "utf8"));
  const presentation = await PresentationFile.importPptx(await FileBlob.load(inputPath));
  const slides = presentation.slides.items;

  if (slides.length !== notes.length) {
    throw new Error(`Slide count mismatch: ppt has ${slides.length} slides, notes file has ${notes.length} items.`);
  }

  for (let index = 0; index < slides.length; index += 1) {
    const slide = slides[index];
    const note = notes[index];
    if (!note?.notes?.trim()) {
      throw new Error(`Missing notes for slide ${index + 1}.`);
    }
    slide.speakerNotes.setText(note.notes.trim());
  }

  const exported = await PresentationFile.exportPptx(presentation);
  await exported.save(outputPath);
  console.log(`Saved notes deck to ${outputPath}`);
}

main().catch((error) => {
  console.error(error?.stack || String(error));
  process.exit(1);
});
