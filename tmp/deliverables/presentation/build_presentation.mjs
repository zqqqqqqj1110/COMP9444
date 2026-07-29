import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";


const ROOT = "D:\\AirSim\\rl_drone_navigation";
const WORK = path.join(ROOT, "tmp", "deliverables", "presentation");
const STARTER = path.join(WORK, "template-starter.pptx");
const OUTPUT = path.join(ROOT, "COMP9444_Presentation_Autonomous_Drone_Navigation.pptx");
const RENDER = path.join(WORK, "final-render");

const reportAssets = path.join(ROOT, "tmp", "deliverables", "report", "assets");
const pptAssets = path.join(WORK, "assets");

let blueprintRecords = [];
let currentRecords = [];


function parseNdjson(text) {
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}


function resolveMapped(presentation, blueprintId) {
  const source = blueprintRecords.find((record) => record.id === blueprintId);
  if (!source) {
    throw new Error(`Blueprint anchor not found: ${blueprintId}`);
  }
  const candidates = currentRecords.filter((record) => {
    if (record.kind !== source.kind || record.slide !== source.slide) return false;
    if (source.kind === "slide") return true;
    return record.name === source.name;
  });
  if (candidates.length !== 1) {
    throw new Error(
      `Could not uniquely map ${blueprintId} (${source.kind}, slide ${source.slide}, ${source.name ?? ""}); matches=${candidates.length}`
    );
  }
  return presentation.resolve(candidates[0].id);
}


async function imageBytes(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}


function setText(presentation, id, text) {
  const target = resolveMapped(presentation, id);
  target.text = text;
  return target;
}


async function replaceImage(presentation, id, imagePath, alt, fit = "cover") {
  const image = resolveMapped(presentation, id);
  const oldFrame = image.frame;
  const oldCrop = image.crop;
  const oldGeometry = image.geometry;
  const oldBorderRadius = image.borderRadius;
  const oldRotation = image.rotation;
  const oldFlipHorizontal = image.flipHorizontal;
  const oldFlipVertical = image.flipVertical;
  const oldLockAspectRatio = image.lockAspectRatio;
  const ext = path.extname(imagePath).toLowerCase();
  const contentType = ext === ".png" ? "image/png" : "image/jpeg";
  image.replace({
    blob: await imageBytes(imagePath),
    contentType,
    alt,
    fit,
  });
  image.frame = oldFrame;
  image.crop = oldCrop;
  image.geometry = oldGeometry;
  image.borderRadius = oldBorderRadius;
  image.rotation = oldRotation;
  image.flipHorizontal = oldFlipHorizontal;
  image.flipVertical = oldFlipVertical;
  image.lockAspectRatio = oldLockAspectRatio;
}


function hideInheritedImage(presentation, id) {
  const image = resolveMapped(presentation, id);
  const hidden = { left: -8, top: -8, width: 1, height: 1 };
  image.position = hidden;
  image.frame = hidden;
}


async function addSlideImage(slide, imagePath, alt, fit = "cover") {
  const ext = path.extname(imagePath).toLowerCase();
  const contentType = ext === ".png" ? "image/png" : "image/jpeg";
  return slide.images.add({
    blob: await imageBytes(imagePath),
    contentType,
    alt,
    fit,
    geometry: "rect",
    position: { left: 552, top: 321, width: 704, height: 297 },
  });
}


function setNotes(presentation, slideId, lines, sources) {
  const slide = resolveMapped(presentation, slideId);
  const noteText = [
    ...lines,
    "",
    "[Sources]",
    ...sources.map((source) => `- ${source}`),
  ].join("\n");
  slide.speakerNotes.textFrame.setText(noteText);
  slide.speakerNotes.setVisible(true);
}


function addNode(slide, name, text, position, fill = "#F4F5F6", line = "#30363D") {
  const box = slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: line, width: 1.2 },
  });
  box.text = text;
  box.text.style = {
    fontSize: 16,
    bold: true,
    color: "#111111",
    horizontalAlignment: "center",
    verticalAlignment: "middle",
  };
  return box;
}


function addArrow(slide, name, position, direction = "down") {
  return slide.shapes.add({
    geometry: direction === "right" ? "rightArrow" : "downArrow",
    name,
    position,
    fill: "#FFD100",
    line: { style: "solid", fill: "#B28A00", width: 0.8 },
  });
}


async function build() {
  await fs.mkdir(RENDER, { recursive: true });
  const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER));
  blueprintRecords = parseNdjson(
    await fs.readFile(path.join(WORK, "template-starter.pptx.inspect.ndjson"), "utf8")
  );
  const currentInspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes",
    maxChars: 100000,
  });
  currentRecords = parseNdjson(currentInspect.ndjson);

  // Slide 1: title.
  setText(presentation, "sh/6107apgn", "Autonomous Drone Navigation\nwith Deep Reinforcement Learning");
  setText(
    presentation,
    "sh/729o3uh8",
    "COMP9444 Project Team\n\nVisual depth | DQN | PPO | Curriculum"
  );
  await replaceImage(
    presentation,
    "im/hcjmdszi",
    path.join(pptAssets, "airsim_success_mid.jpg"),
    "AirSimNH drone view with depth inset"
  );
  await replaceImage(
    presentation,
    "im/gba547ix",
    path.join(reportAssets, "final_outcomes.png"),
    "Final deterministic outcome comparison",
    "contain"
  );
  setNotes(
    presentation,
    "sl/t2f18f",
    [
      "Introduce the project as visual point-to-point navigation without a map.",
      "State the comparison: vanilla DQN, PPO from scratch, and PPO curriculum.",
    ],
    [
      path.join(ROOT, "README.md"),
      path.join(ROOT, "experiments", "airsimnh", "validated_comparison_seed7_test_seed20007.csv"),
    ]
  );

  // Slide 2: motivation and problem.
  setText(presentation, "sh/y9kfmpwr", "Depth replaces a stored obstacle map");
  const body2 = setText(
    presentation,
    "sh/zatgvudc",
    [
      "Goal: travel 32.55 m through AirSimNH and stop within 2 m",
      "No map or explicit obstacle coordinates",
      "Input: front depth + relative goal + velocity",
      "Six discrete body-frame actions every 0.35 s",
      "Safety failures terminate the episode",
    ].join("\n")
  );
  body2.position = { left: 36, top: 112, width: 500, height: 505 };
  setText(presentation, "sh/8f2hsju9", "2");
  setText(presentation, "sh/90vy14ve", "Presenter: COMP9444 Project Team");
  hideInheritedImage(presentation, "im/fydkz2x4");
  await addSlideImage(
    resolveMapped(presentation, "sl/gk5gr3"),
    path.join(pptAssets, "airsim_collision_mid.jpg"),
    "AirSimNH obstacle encounter with policy depth input"
  );
  setNotes(
    presentation,
    "sl/gk5gr3",
    [
      "Explain that the route contains a house that blocks direct goal pursuit.",
      "The policy must infer free space from depth while still following relative target direction.",
    ],
    [
      path.join(ROOT, "src", "airsim_drone_env.py"),
      path.join(ROOT, "experiments", "airsimnh", "ppo", "curriculum_stage02_23m_10k_seed7_stable_v2_stage2_pilot", "recordings"),
    ]
  );

  // Slide 3: literature review I.
  setText(presentation, "sh/ze9wva54", "AirSim enables learning; DQN is the baseline");
  setText(
    presentation,
    "sh/ed0fm54z",
    [
      "AirSim [1]",
      "Unreal Engine visuals, multirotor physics, cameras and collision queries",
      "Repeatable interaction without risking physical hardware",
      "Deep Q-Network [2]",
      "Learns discrete action values from image features",
      "Replay buffer + target network improve training stability",
      "Our baseline deliberately remains vanilla DQN",
    ].join("\n")
  );
  setText(presentation, "sh/dcretkne", "3");
  setText(presentation, "sh/cbyxkfmt", "Presenter: COMP9444 Project Team");
  setNotes(
    presentation,
    "sl/k4ux7i",
    [
      "Delineate prior work from our implementation.",
      "AirSim and DQN are established foundations; our contribution is the navigation environment and controlled comparison.",
    ],
    [
      "https://arxiv.org/abs/1705.05065",
      "https://www.nature.com/articles/nature14236",
    ]
  );

  // Slide 4: literature review II.
  setText(presentation, "sh/eh0zypg7", "PPO limits updates; curriculum orders tasks");
  setText(
    presentation,
    "sh/zit0ruxc",
    [
      "Proximal Policy Optimisation [3]",
      "Clipped surrogate objective limits destructive updates",
      "Actor-critic learning supports stochastic discrete control",
      "Curriculum learning [4]",
      "Orders easier tasks before harder tasks",
      "Our curriculum increases target distance: 10 m -> 23 m -> 33 m",
      "Question: does curriculum help under the same 45k interactions?",
    ].join("\n")
  );
  setText(presentation, "sh/on2h4zy9", "4");
  setText(presentation, "sh/9obix4fu", "Presenter: COMP9444 Project Team");
  setNotes(
    presentation,
    "sl/j1docj",
    [
      "PPO is selected because the action space is discrete and policy stability matters.",
      "Curriculum is a hypothesis, not an assumed improvement; the experiment tests it.",
    ],
    [
      "https://arxiv.org/abs/1707.06347",
      "https://icml.cc/2009/papers/119.pdf",
    ]
  );

  // Slide 5: RL task and exploratory analysis.
  setText(presentation, "sh/fqlsnitg", "AirSim supplies online experience");
  const body5 = setText(
    presentation,
    "sh/epcbudsv",
    [
      "Observation",
      "84 x 84 front depth, target offset and velocity",
      "Pre-processing",
      "Clip depth at 40 m; resize; convert near obstacles to high intensity",
      "Challenge",
      "Sparse +100 goal reward, unsafe exploration, partial observability",
      "No classes, train/test images, SLAM or obstacle labels",
    ].join("\n")
  );
  body5.position = { left: 36, top: 112, width: 500, height: 505 };
  setText(presentation, "sh/dojalsba", "5");
  setText(presentation, "sh/snatsnap", "Presenter: COMP9444 Project Team");
  hideInheritedImage(presentation, "im/y14zuhc7");
  await addSlideImage(
    resolveMapped(presentation, "sl/mexs9p"),
    path.join(pptAssets, "airsim_obstacle_view.jpg"),
    "AirSim camera frame and corresponding depth input"
  );
  setNotes(
    presentation,
    "sl/mexs9p",
    [
      "Relate this slide directly to the rubric's data or RL task analysis.",
      "The depth inset is the policy input; RGB is recorded only for human inspection.",
    ],
    [
      path.join(ROOT, "src", "airsim_drone_env.py"),
      path.join(ROOT, "experiments", "airsimnh", "ppo", "curriculum_stage02_23m_10k_seed7_stable_v2_stage2_pilot", "recordings"),
    ]
  );

  // Slide 6: architecture.
  setText(presentation, "sh/retw32ho", "A shared CNN keeps the algorithm comparison controlled");
  const body6 = setText(
    presentation,
    "sh/6tkvaxgj",
    [
      "Common perception",
      "3 convolution layers -> 3,136 visual features",
      "DQN",
      "Replay + epsilon-greedy -> six Q-values",
      "Stable PPO",
      "LayerNorm, Tanh, orthogonal initialisation",
      "Actor logits + critic value",
    ].join("\n")
  );
  body6.position = { left: 36, top: 124, width: 610, height: 490 };
  setText(presentation, "sh/5sbe1szy", "6");
  setText(presentation, "sh/jqtwzihs", "Presenter: COMP9444 Project Team");
  const slide6 = resolveMapped(presentation, "sl/ghy4w1");
  addArrow(slide6, "architecture-arrow-1", { left: 800, top: 230, width: 56, height: 34 });
  addArrow(slide6, "architecture-arrow-2", { left: 800, top: 350, width: 56, height: 34 });
  addArrow(slide6, "architecture-arrow-3", { left: 745, top: 468, width: 46, height: 28 }, "right");
  addArrow(slide6, "architecture-arrow-4", { left: 1004, top: 468, width: 46, height: 28 }, "right");
  addNode(slide6, "depth-node", "84 x 84\nDepth", { left: 690, top: 145, width: 205, height: 80 }, "#E9F4F1");
  addNode(slide6, "state-node", "6-D Goal\n+ Velocity", { left: 980, top: 145, width: 205, height: 80 }, "#FFF4D6");
  addNode(slide6, "fusion-node", "CNN 3,136 + State 6\n= 3,142 features", { left: 755, top: 270, width: 370, height: 76 }, "#F4F5F6");
  addNode(slide6, "shared-node", "FC 512 -> LayerNorm -> Tanh", { left: 755, top: 390, width: 370, height: 70 }, "#F4F5F6");
  addNode(slide6, "actor-node", "Actor\n6 logits", { left: 680, top: 505, width: 210, height: 64 }, "#E8F0FA");
  addNode(slide6, "critic-node", "Critic\nV(s)", { left: 980, top: 505, width: 210, height: 64 }, "#FCECE8");
  setNotes(
    presentation,
    "sl/ghy4w1",
    [
      "Emphasise that DQN and PPO see the same representation and action set.",
      "Layer normalisation was added before Tanh after diagnosing hidden-feature saturation.",
    ],
    [
      path.join(ROOT, "src", "dqn_agent.py"),
      path.join(ROOT, "src", "ppo_agent.py"),
    ]
  );

  // Slide 7: experimental protocol.
  setText(presentation, "sh/cf6honi5", "Validation selects; a fresh seed tests once");
  const body7 = setText(
    presentation,
    "sh/dgfixszq",
    [
      "Fair budget",
      "45,000 interactions per method",
      "Same final task",
      "Start, target, reward and max 150 steps",
      "Metrics",
      "Success, collision, timeout, reward",
      "Steps, path length, final distance, min depth",
      "Training seed: 7 | Test seed: 20007",
    ].join("\n")
  );
  body7.position = { left: 36, top: 124, width: 620, height: 490 };
  setText(presentation, "sh/29ozid03", "7");
  setText(presentation, "sh/4b6hkni9", "Presenter: COMP9444 Project Team");
  const slide7 = resolveMapped(presentation, "sl/voirjg");
  addArrow(slide7, "protocol-arrow-1", { left: 908, top: 265, width: 54, height: 35 });
  addArrow(slide7, "protocol-arrow-2", { left: 908, top: 405, width: 54, height: 35 });
  addNode(slide7, "protocol-stage-1", "All checkpoints\n5 episodes each\nSeed 7", { left: 735, top: 160, width: 400, height: 95 }, "#FFF4D6");
  addNode(slide7, "protocol-stage-2", "Top 3 checkpoints\n30 episodes each\nSeed 10007", { left: 735, top: 305, width: 400, height: 95 }, "#E8F0FA");
  addNode(slide7, "protocol-stage-3", "Selected model\n50 final-test episodes\nSeed 20007", { left: 735, top: 445, width: 400, height: 95 }, "#E9F4F1");
  setNotes(
    presentation,
    "sl/voirjg",
    [
      "Explain why selection and final testing are separated.",
      "The test seed is independent, but the route geometry remains fixed.",
    ],
    [
      path.join(ROOT, "run_checkpoint_selection.ps1"),
      path.join(ROOT, "experiments", "airsimnh", "validated_comparison_seed7_test_seed20007.csv"),
    ]
  );

  // Slide 8: training behaviour.
  setText(presentation, "sh/upszed0n", "Scratch PPO improves late; curriculum lags");
  const body8 = setText(
    presentation,
    "sh/fql07ih8",
    [
      "Scratch PPO reaches 0.7-0.8 moving success late in training",
      "DQN remains mostly near zero despite epsilon exploration",
      "Curriculum changes the target at 5k and 15k interactions",
      "Easy-stage success is not final-route success",
      "All curves remain noisy: checkpoint selection matters",
    ].join("\n")
  );
  body8.position = { left: 36, top: 112, width: 500, height: 505 };
  setText(presentation, "sh/snahc3ih", "8");
  setText(presentation, "sh/toji58z2", "Presenter: COMP9444 Project Team");
  hideInheritedImage(presentation, "im/3edk7i5o");
  await addSlideImage(
    resolveMapped(presentation, "sl/b613ut"),
    path.join(reportAssets, "training_success.png"),
    "Training success moving averages by consumed interactions",
    "contain"
  );
  setNotes(
    presentation,
    "sl/b613ut",
    [
      "Do not compare raw curriculum rewards across stages because route lengths differ.",
      "The plot uses 20-episode moving success and marks curriculum target changes.",
    ],
    [
      path.join(ROOT, "experiments", "airsimnh", "dqn", "scratch_33m_45k_seed7_stable_v3_scratch", "results", "training_log.csv"),
      path.join(ROOT, "experiments", "airsimnh", "ppo", "scratch_33m_45k_seed7_stable_v3_scratch", "results", "training_log.csv"),
    ]
  );

  // Slide 9: final results.
  setText(presentation, "sh/50zelo3i", "PPO Scratch reaches 98% success");
  const body9 = setText(
    presentation,
    "sh/4zqds3mx",
    [
      "PPO Scratch",
      "98% success | 2% collision | 0% timeout",
      "Selected checkpoint",
      "42,500 interactions",
      "Efficiency",
      "52.28 steps | 38.27 m path",
      "Mean reward: 141.1",
    ].join("\n")
  );
  body9.position = { left: 36, top: 130, width: 560, height: 470 };
  setText(presentation, "sh/vuxcfelg", "9");
  setText(presentation, "sh/hwfuho3m", "Presenter: COMP9444 Project Team");
  const slide9 = resolveMapped(presentation, "sl/v73fvt");
  slide9.charts.add("bar", {
    position: { left: 635, top: 150, width: 590, height: 430 },
    categories: ["DQN Scratch", "PPO Scratch", "PPO Curriculum"],
    series: [
      { name: "Success", values: [16, 98, 68], fill: "#178F74" },
      { name: "Collision", values: [72, 2, 2], fill: "#C44536" },
      { name: "Timeout", values: [14, 0, 30], fill: "#D88C00" },
    ],
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 13 } },
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 75 },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 13 } },
    xAxis: { textStyle: { fontSize: 13 }, line: { style: "solid", fill: "#707780", width: 1 } },
    yAxis: {
      min: 0,
      max: 100,
      majorUnit: 20,
      numberFormatCode: '0"%"',
      textStyle: { fontSize: 12 },
      majorGridlines: { style: "solid", fill: "#D7DCE1", width: 1 },
    },
    chartFill: "#FFFFFF",
    chartLine: { style: "solid", fill: "none", width: 0 },
    plotAreaFill: "#FFFFFF",
    plotAreaLine: { style: "solid", fill: "none", width: 0 },
  });
  setNotes(
    presentation,
    "sl/v73fvt",
    [
      "This is the primary deterministic comparison from 50 final-test episodes.",
      "PPO Scratch succeeds in 49/50 episodes; the single failure is a collision.",
    ],
    [
      path.join(ROOT, "experiments", "airsimnh", "validated_comparison_seed7_test_seed20007.csv"),
    ]
  );

  // Slide 10: error analysis and policy mode.
  setText(presentation, "sh/0361gzmx", "The three policies fail in different ways");
  const body10 = setText(
    presentation,
    "sh/l4f2p43i",
    [
      "DQN: direct pursuit and 72% collision; mean min depth 0.37 m",
      "Curriculum PPO: only 2% collision, but 30% timeout",
      "Scratch PPO learns a repeatable lateral detour",
      "Sampling exposes residual uncertainty",
      "PPO Scratch: 98% deterministic -> 66% stochastic",
      "PPO Curriculum: 68% deterministic -> 46% stochastic",
    ].join("\n")
  );
  body10.position = { left: 36, top: 112, width: 500, height: 505 };
  setText(presentation, "sh/qd4ja54v", "10");
  setText(presentation, "sh/bedkjalg", "Presenter: COMP9444 Project Team");
  hideInheritedImage(presentation, "im/9kb6h8fi");
  await addSlideImage(
    resolveMapped(presentation, "sl/ykb6ef"),
    path.join(reportAssets, "representative_trajectories.png"),
    "Representative deterministic trajectories for DQN and PPO",
    "contain"
  );
  setNotes(
    presentation,
    "sl/ykb6ef",
    [
      "The trajectory figure uses deterministic Episode 1 from each selected model.",
      "Explain that deterministic is the deployment policy; stochastic sampling is a diagnostic.",
    ],
    [
      path.join(ROOT, "experiments", "airsimnh", "validated_comparison_seed7_test_seed20007.csv"),
      path.join(ROOT, "experiments", "airsimnh", "ppo", "scratch_33m_45k_seed7_stable_v3_scratch_validated_test_seed20007", "results", "evaluation_deterministic_trajectory.csv"),
    ]
  );

  // Slide 11: conclusion and limitations.
  setText(presentation, "sh/1gbq10zi", "Strong on one route; generalisation remains open");
  setText(
    presentation,
    "sh/gf2p8fix",
    [
      "What worked",
      "Stable PPO learned a safe, efficient detour: 98% success and 2% collision",
      "What did not",
      "Vanilla DQN collided; curriculum PPO preserved a short-horizon timeout bias",
      "What we can claim",
      "Strong fixed-route AirSimNH performance under a controlled budget",
      "What we cannot claim",
      "Generalisation to unseen routes, scenes or real drones",
      "Next: 3-5 seeds, route perturbations, second scene, multi-route training",
    ].join("\n")
  );
  setText(presentation, "sh/ra9ovqhg", "11");
  setText(presentation, "sh/tcr6x0z6", "Presenter: COMP9444 Project Team");
  setNotes(
    presentation,
    "sl/lzpx68",
    [
      "Close by separating evidence from future hypotheses.",
      "The recommended inference model is pretrained/airsimnh/ppo_scratch_seed7.pt.",
    ],
    [
      path.join(ROOT, "README.md"),
      path.join(ROOT, "pretrained", "airsimnh", "ppo_scratch_seed7.pt"),
    ]
  );

  // Slide 12: questions.
  setText(presentation, "sh/zqhs7qpw", "Questions");
  setText(presentation, "sh/9wjadk7y", "12");
  const license = resolveMapped(presentation, "sh/p0za1gre");
  license.text = "";
  await replaceImage(
    presentation,
    "im/x0behgze",
    path.join(pptAssets, "airsim_goal_approach.jpg"),
    "AirSimNH goal approach with depth input"
  );
  setNotes(
    presentation,
    "sl/alhl8a",
    [
      "Invite questions on reward design, checkpoint selection, policy modes, and generalisation.",
    ],
    [
      path.join(ROOT, "experiments", "airsimnh", "ppo", "curriculum_stage02_23m_10k_seed7_stable_v2_stage2_pilot", "recordings"),
    ]
  );

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(RENDER, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(RENDER, `${stem}.layout.json`), await layout.text());
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(RENDER, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const inspect = await presentation.inspect({
    kind: "slide,textbox,shape,image,table,chart,notes",
    maxChars: 30000,
  });
  await fs.writeFile(path.join(RENDER, "final-inspect.ndjson"), inspect.ndjson, "utf8");

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT);
  console.log(OUTPUT);
}


build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
