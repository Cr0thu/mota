#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repoRoot = path.resolve(__dirname, "..");
const defaultProject = path.join(
  repoRoot,
  "game",
  "Falsh原版魔塔合集",
  "51_2",
  "project",
);
const projectDir = process.argv[2] ? path.resolve(process.argv[2]) : defaultProject;
const outPath = process.argv[3]
  ? path.resolve(process.argv[3])
  : path.join(repoRoot, "artifacts", "data", "mota_first10.json");

function runFile(context, filePath) {
  const source = fs.readFileSync(filePath, "utf8");
  vm.runInContext(source, context, { filename: filePath });
}

function resolveVar(context, prefix) {
  const key = Object.keys(context).find((name) => name.startsWith(prefix));
  if (!key) throw new Error(`Could not find variable with prefix ${prefix}`);
  return context[key];
}

function main() {
  const context = {
    main: { floors: {} },
    console,
  };
  vm.createContext(context);

  for (const name of ["data.js", "maps.js", "enemys.js", "items.js"]) {
    runFile(context, path.join(projectDir, name));
  }
  for (let i = 0; i <= 10; i += 1) {
    runFile(context, path.join(projectDir, "floors", `MT${i}.js`));
  }

  const data = resolveVar(context, "data_");
  const maps = resolveVar(context, "maps_");
  const enemys = resolveVar(context, "enemys_");
  const items = resolveVar(context, "items_");

  const floors = {};
  for (let i = 0; i <= 10; i += 1) {
    floors[`MT${i}`] = context.main.floors[`MT${i}`];
  }

  const payload = {
    source_project: projectDir,
    extracted_at: new Date().toISOString(),
    firstData: data.firstData,
    values: data.values,
    flags: data.flags,
    maps,
    enemys,
    items,
    floors,
  };

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
  console.log(outPath);
}

main();

