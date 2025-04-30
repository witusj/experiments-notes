# **A Hands-On Tutorial: Discrete Bayesian Optimization for Patient Scheduling with , JS & Tailwind**

## **1. Introduction: Smarter Patient Scheduling with Discrete Bayesian Optimization**

### **1.1 The Patient Scheduling Puzzle**

Healthcare providers constantly face the complex challenge of patient scheduling. It involves juggling multiple patients, each requiring different appointment types and durations, with limited resources like doctors, nurses, and examination rooms, all within constrained time slots. The goal is often multifaceted: minimize patient waiting times, maximize the utilization of valuable resources, accommodate patient preferences, and ensure smooth clinic flow. Manually creating optimal schedules is difficult, and simple rule-based systems often struggle as the number of patients, resources, and constraints increases. Finding the *absolute best* schedule becomes a computationally demanding puzzle, setting the stage for more sophisticated optimization techniques.

### **1.2 Introducing Bayesian Optimization (BO): Learning to Optimize Efficiently**

Bayesian Optimization (BO) offers a powerful, data-efficient approach to tackling such complex optimization problems. It's a sequential design strategy particularly well-suited for optimizing "black-box" functions – situations where we don't have a neat mathematical formula for the objective (like schedule quality) and evaluating it is expensive or time-consuming (e.g., running a detailed clinic simulation or observing real-world outcomes).1

Think of BO like a skilled physician trying to find the best treatment dosage for a patient. The physician doesn't test every possible dose randomly. Instead, they start with an initial guess, observe the patient's response, update their understanding of how the dosage affects the patient, and then intelligently choose the next dosage to try based on that updated understanding. Similarly, BO uses the results from previously evaluated configurations (schedules, in our case) to build a probabilistic model of the objective function and then uses this model to decide the most promising configuration to evaluate next.3 This iterative, model-guided approach allows BO to find good solutions with significantly fewer evaluations compared to exhaustive search or random sampling.3 Its efficiency has led to successful applications in diverse fields like tuning complex machine learning models 6, robotics 3, materials discovery 8, and optimizing experimental designs.10

### **1.3 Why *Discrete* BO? Handling Choices, Not Just Knobs**

While standard BO often deals with tuning continuous parameters (like adjusting temperature or pressure smoothly), many real-world problems, including patient scheduling, involve making *discrete choices*. We need to assign Patient A to Slot 1 *or* Slot 2, use Room X *or* Room Y, assign Doctor Z *or* Nurse Y. These are distinct, separate options, not points on a continuous scale.

This is where Discrete Bayesian Optimization (DBO) comes in. DBO methods are specifically designed to handle optimization problems where the decision variables belong to a discrete set.1 Trying to adapt continuous BO methods, for instance, by suggesting a continuous slot "1.7" and rounding it to the nearest integer slot "2", can be problematic. This rounding approach might cause the optimizer to repeatedly suggest points already evaluated or get stuck, failing to explore the discrete space effectively.12 DBO directly operates on the discrete set of choices, making it a more natural fit for problems like selecting optimal compounds from a chemical library 8, protein engineering 1, or, in our case, assigning patients to specific time slots.

The need for specialized discrete methods arises because the structure of discrete spaces is fundamentally different from continuous ones. Techniques that rely on smooth gradients or continuous proximity don't directly apply. DBO algorithms often involve building models appropriate for discrete inputs and adapting acquisition strategies to select the best discrete candidate from the available options.8

### **1.4 Tutorial Goal & Structure**

The goal of this tutorial is to provide a hands-on introduction to the core concepts of Discrete Bayesian Optimization by building a simple web-based simulator. Using plain , JavaScript, and Tailwind CSS, we will create an interactive tool that applies DBO principles to a simplified patient scheduling problem.

The tutorial is structured as follows:

1.  **Understanding the Core Ideas:** We'll break down the BO loop, surrogate models, acquisition functions, and how they adapt to discrete choices.\
2.  **Building the Simulator Interface:** We'll set up the  structure and use Tailwind CSS for styling.\
3.  **Building the Simulator Logic:** We'll implement the DBO components (objective function, simplified surrogate, simplified acquisition function, main loop) in JavaScript.\
4.  **Running and Exploring:** We'll provide the complete code, instructions to run it, discuss the necessary simplifications made, and suggest avenues for further learning.

This tutorial aims to build intuition and practical understanding of DBO concepts within an estimated study time of approximately 3 hours. It serves as an educational stepping stone, not a production-ready scheduling system.

## **2. Understanding the Core Ideas**

Before diving into the code, let's clarify the fundamental components of Bayesian Optimization, particularly in the context of discrete choices.

### **2.1 The Bayesian Optimization Loop Explained Simply**

Bayesian Optimization works iteratively, continuously refining its understanding of the problem and making increasingly informed decisions. The core loop generally follows these steps 3:

1.  **Initial Evaluation(s):** Start by evaluating the objective function for one or more initial configurations (schedules). This provides the first data points for the model. In some cases, these initial points might be chosen randomly or based on prior knowledge.\
2.  **Build/Update Surrogate Model:** Use all the evaluated configurations and their corresponding objective function scores gathered so far to build or update a statistical model. This "surrogate model" acts as an inexpensive approximation of the true, expensive-to-evaluate objective function.3 It captures our current belief about how different configurations affect the outcome.\
3.  **Optimize Acquisition Function:** Based on the predictions and uncertainty estimates from the surrogate model, calculate an "acquisition function" score for potential unevaluated configurations. This function quantifies the utility or desirability of evaluating each candidate next.8 Select the configuration that maximizes this acquisition function – this is the point deemed most promising for evaluation in the next iteration.4\
4.  **Evaluate Chosen Configuration:** Evaluate the actual objective function for the configuration selected in the previous step. This yields a new data point (configuration, score).\
5.  **Augment Data & Repeat:** Add the new data point to the set of observations and return to Step 2 to update the surrogate model with the new information. This cycle continues until a stopping criterion is met (e.g., a maximum number of evaluations is reached, a satisfactory objective value is achieved, or the potential for further improvement diminishes).8

This iterative process allows BO to intelligently explore the search space, focusing evaluations where they are most likely to yield improvements or reduce uncertainty, leading to efficient optimization.3

**(Video Embedding - Conceptual)**

To visualize this loop, consider watching an introductory video on Bayesian Optimization. Videos like the one from Taylor Sparks 16 or overviews discussing BO for experimental design 10 can provide a helpful conceptual grounding. They often illustrate how the model refines over iterations and guides the search.



<div class="my-4">\
<iframe width="560" height="315" src="https://www.youtube.com/embed/PrEuA8hm9mY" title="YouTube video player - Bayesian Optimization Introduction" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\
<p class="text-sm text-gray-600 mt-1">Video discussing Bayesian Optimization concepts and heuristics.\[16, 17, 18, 19\]</p>\
</div>

### **2.2 Surrogate Models: Our "Smart Guess" for Schedule Quality**

The surrogate model is the heart of Bayesian Optimization. Since the true objective function (e.g., schedule quality) is unknown and expensive to evaluate, the surrogate model acts as a cheap, stand-in approximation based on the data observed so far.3 Its key roles are:

-   **Prediction:** It provides a prediction (often called the mean prediction) of the objective function's value for any given configuration, even those not yet evaluated.\
-   **Uncertainty Quantification:** Crucially, it also provides a measure of uncertainty (often variance or standard deviation) associated with its predictions. Predictions in regions with many nearby observations will typically have low uncertainty, while predictions in unexplored regions will have high uncertainty.3

This combination of prediction and uncertainty allows the acquisition function (discussed next) to make intelligent decisions about where to sample next.

Common types of surrogate models include:

-   **Gaussian Processes (GPs):** GPs are arguably the most popular surrogate model for BO.3 They define a probability distribution over functions, allowing them to model complex relationships and provide well-calibrated uncertainty estimates.12 Conceptually, a GP fits a flexible curve (in 1D) or surface (in higher dimensions) through the observed data points, along with "confidence bands" representing uncertainty.3 They typically use a kernel function (like Matern or RBF) to define the smoothness and correlation between points.3 However, standard GPs can become computationally expensive as the number of observations grows, and adapting them effectively to purely discrete or high-dimensional discrete spaces can require specialized techniques.1 Implementing GPs involves matrix operations (like inverting the kernel matrix) which can be complex.1\
-   **Random Forests (RFs):** A Random Forest is an ensemble method that builds multiple decision trees on different subsets of the data and features, averaging their predictions.24 RFs can naturally handle discrete and categorical input variables and are often computationally faster to train than GPs.24 While primarily used for classification and regression, they can also serve as surrogate models in optimization.12 Estimating predictive uncertainty with RFs is possible but sometimes considered less direct or principled than with GPs.12

The choice of surrogate model depends on the nature of the problem (continuous vs. discrete inputs, expected smoothness, dimensionality) and computational constraints. For this tutorial, implementing a full GP or RF in JavaScript is impractical due to complexity and the time limit. We will therefore use a highly simplified approach that captures the *essence* of a surrogate: storing observations and providing basic prediction and uncertainty estimates, allowing us to focus on the overall DBO loop structure. This simplification is a key adaptation required to bridge the gap between the advanced methods often discussed in research 1 and a practical, introductory implementation in JavaScript.

### **2.3 Acquisition Functions: Deciding Which Schedule to Try Next**

Once the surrogate model provides predictions and uncertainty estimates, the **acquisition function** uses this information to decide which unevaluated point (schedule) is the most valuable to evaluate next.3 It essentially translates the surrogate's probabilistic forecast into a score that quantifies the "utility" of sampling each potential point. The point with the highest acquisition score is chosen for the next expensive evaluation.4

The core challenge for an acquisition function is to balance two competing goals 4:

-   **Exploitation:** Focusing on regions where the surrogate model predicts good objective function values (based on the mean prediction). This aims to refine the solution in already promising areas.\
-   **Exploration:** Investigating regions where the surrogate model is highly uncertain (high variance/standard deviation). This aims to discover potentially better regions that haven't been explored yet and improve the global accuracy of the surrogate model.

Several popular acquisition functions exist, each balancing this trade-off differently:

-   **Expected Improvement (EI):** EI calculates the expected *amount* by which the objective function value at a point x will exceed the current best observed value, fbest​. It considers both the predicted mean μ(x) and standard deviation σ(x) from the surrogate model.20 Points with high predicted means *or* high uncertainty can have high EI scores, making it a popular and well-balanced choice.3 The calculation involves the probability density function (ϕ) and cumulative distribution function (Φ) of the standard normal distribution.28\
-   **Probability of Improvement (PI):** PI calculates the *probability* that the objective function value at a point x will be better than the current best observed value fbest​.14 It primarily focuses on the likelihood of making *any* improvement, regardless of the magnitude.14 This can sometimes lead it to be more exploitative than EI, potentially getting stuck in local optima if not carefully managed.14\
-   **Upper Confidence Bound (UCB):** UCB takes a more direct approach to balancing the trade-off. It calculates a score based on an optimistic estimate of the objective function value: UCB(x)=μ(x)+βσ(x), where μ(x) is the predicted mean, σ(x) is the predicted standard deviation, and β is a tunable parameter that controls the emphasis on exploration (higher β favors exploration).14 Points with high means or high uncertainty receive high UCB scores. It's often considered effective for encouraging exploration.30

Just like the surrogate model, implementing the exact mathematical formulas for these acquisition functions (especially EI, which involves Gaussian probability functions) can be complex, particularly when not using a standard GP surrogate. Given the simplified surrogate we'll use in this JavaScript tutorial, we will implement a simplified acquisition logic inspired by UCB. The UCB structure (mean+β×uncertainty) is conceptually straightforward to adapt using our simplified prediction and uncertainty estimates. This allows us to demonstrate the core exploration-exploitation mechanism without getting bogged down in complex probability calculations, making it suitable for our learning objectives and constraints.

**(Video Embedding - Conceptual)**

Videos discussing acquisition functions, particularly the exploration vs. exploitation trade-off or specific functions like UCB/LCB (Lower Confidence Bound, used for minimization), can enhance understanding.



<div class="my-4">\
<iframe width="560" height="315" src="https://www.youtube.com/embed/\_SC5_2vkgbA" title="YouTube video player - Acquisition Functions (LCB Example)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\
<p class="text-sm text-gray-600 mt-1">Video discussing acquisition functions like Lower Confidence Bound (LCB), related to UCB.\[18\]</p>\
</div>

### **2.4 Handling Discrete Choices**

Applying these concepts to our patient scheduling problem requires acknowledging its discrete nature. We are not searching over a continuous range of parameters but selecting from a finite (though potentially large) set of possible schedule assignments.1

Key considerations for DBO include:

-   **Direct Operation:** DBO algorithms work directly on the discrete search space. They don't rely on continuous approximations or rounding, which can be inefficient or ineffective.12\
-   **Surrogate Model Suitability:** The surrogate model should ideally be capable of handling discrete inputs effectively. While GPs can be adapted (e.g., using specific kernels or embeddings), models like Random Forests might handle categorical or discrete features more naturally.1 Our simplified surrogate will inherently work with the discrete schedule representations.\
-   **Acquisition Function Optimization:** In continuous BO, finding the maximum of the acquisition function often requires numerical optimization techniques.2 In DBO, especially when the number of discrete candidates is manageable, optimizing the acquisition function can be done simply by evaluating it for all (or a representative subset of) unevaluated discrete candidates and picking the best one.11 This is the approach we will take in our simulator.

## **3. Building the Simulator: Part 1 - The Interface ( & Tailwind)**

Now, let's start building the user interface for our DBO Patient Scheduling simulator.

### **3.1 Defining a Simple Scheduling Problem**

To keep the implementation manageable within the tutorial's scope, we'll define a very simple scheduling scenario:

-   **Patients:** 5 patients (P1 to P5).\
-   **Time Slots:** 10 available time slots (Slot 0 to Slot 9). Assume each slot is long enough for any patient.\
-   **Resources:** 1 doctor (meaning only one patient can be scheduled per slot).\
-   **Schedule Representation:** A simple JavaScript array of length 5. The index represents the patient (0 for P1, 1 for P2, etc.), and the value represents the assigned time slot (0-9). A value of null or -1 could indicate the patient is not scheduled. Example: \[2, 0, -1, 5, 1\] means P1 in Slot 2, P2 in Slot 0, P3 unscheduled, P4 in Slot 5, P5 in Slot 1.\
-   **Constraint:** No two patients can be assigned to the same slot.\
-   **Objective:** Maximize a "schedule quality score". For simplicity, let's define this score based on minimizing the sum of assigned slot numbers (representing earlier appointments being better) and penalizing unscheduled patients. A higher score will be better. (This avoids negative numbers and fits the UCB maximization framework more intuitively than minimizing waiting time directly).

This setup provides a discrete search space (different valid assignments of patients to slots) and a quantifiable objective to optimize.

### **3.2  Structure**

Create an index. file. We'll set up the basic structure to display the optimization progress and allow user interaction.



<!DOCTYPE ****>\
< lang="en">\
<head>\
<meta charset="UTF-8">\
<meta name="viewport" content="width=device-width, initial-scale=1.0">\
<title>DBO Patient Scheduling Simulator</title>\
<script src="https://cdn.tailwindcss.com"></script>\
<style>\
/\* Optional: Add custom base styles or component styles here \*/\
.slot {\
border: 1px solid #ccc;\
padding: 0.5rem;\
margin: 0.2rem;\
min-width: 60px; /\* Adjust as needed \*/\
text-align: center;\
}\
</style>\
</head>\
<body class="bg-gray-100 font-sans p-8">

```         
<div class\="container mx-auto bg-white p-6 rounded-lg shadow-md">  
    <h1 class\="text-2xl font-bold mb-4 text-center text-blue-700">Discrete Bayesian Optimization for Patient Scheduling</h1>

    <div class\="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">  
        <div>  
            <h2 class\="text-xl font-semibold mb-2 text-gray-800">Controls</h2>  
            <button id\="run-iteration-btn" class\="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition duration-150 ease-in-out">  
                Run One DBO Iteration  
            </button>  
            <p class\="text-sm text-gray-600 mt-2">Click to perform one step of the optimization.</p>  
             <div class\="mt-4">  
                <label for\="beta-slider" class\="block text-sm font-medium text-gray-700">Exploration Factor (β): <span id\="beta-value">1.0</span></label>  
                <input type\="range" id\="beta-slider" name\="beta" min\="0.1" max\="5.0" step\="0.1" value\="1.0" class\="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">  
            </div>  
        </div>

        <div>  
            <h2 class\="text-xl font-semibold mb-2 text-gray-800">Best Schedule Found</h2>  
            <p class\="text-lg">Score: <span id\="best-score" class\="font-mono font-bold text-green-600">\-</span></p>  
            <p class\="text-md mt-1">Schedule: <span id\="best-schedule" class\="font-mono text-purple-700">\-</span></p>  
            <p class\="text-sm text-gray-600 mt-2">(Higher score is better)</p>  
        </div>  
    </div>

     <div class\="mb-6">  
        <h2 class\="text-xl font-semibold mb-2 text-gray-800">Schedule Visualization (Best Found)</h2>  
        <div id\="schedule-visualization" class\="flex flex-wrap bg-gray-50 p-2 rounded border border-gray-300">  
            <span class\="text-gray-500 italic">Run an iteration to visualize...</span>  
        </div>  
    </div>

    <div>  
        <h2 class\="text-xl font-semibold mb-2 text-gray-800">Evaluation History</h2>  
        <div class\="overflow-x-auto">  
            <table class\="min-w-full bg-white border border-gray-300">  
                <thead class\="bg-gray-200">  
                    <tr>  
                        <th class\="py-2 px-4 border-b text-left text-sm font-medium text-gray-600">Iteration</th>  
                        <th class\="py-2 px-4 border-b text-left text-sm font-medium text-gray-600">Schedule Evaluated</th>  
                        <th class\="py-2 px-4 border-b text-left text-sm font-medium text-gray-600">Objective Score</th>  
                        <th class\="py-2 px-4 border-b text-left text-sm font-medium text-gray-600">Acquisition Score</th>  
                        <th class\="py-2 px-4 border-b text-left text-sm font-medium text-gray-600">Prediction (Mean)</th>  
                        <th class\="py-2 px-4 border-b text-left text-sm font-medium text-gray-600">Uncertainty (Proxy)</th>  
                    </tr>  
                </thead>  
                <tbody id\="history-table-body">  
                    <tr>  
                        <td colspan\="6" class\="py-2 px-4 border-b text-center text-gray-500 italic">No evaluations yet.</td>  
                    </tr>  
                </tbody>  
            </table>  
        </div>  
    </div>

</div>

<script src\="script.js"></script>  
```

</body>\
</>

This  sets up distinct sections for controls, displaying the best result, visualizing the schedule, and logging the history of evaluated schedules. It includes placeholders (like id="best-score") that our JavaScript will target to display dynamic information.

### **3.3 Styling with Tailwind CSS**

We've included Tailwind via a CDN link in the <head>. The  structure already uses Tailwind's utility classes for layout, spacing, typography, colors, and basic styling:

-   **Layout:** container, grid, grid-cols-1, md:grid-cols-2, gap-6, flex, flex-wrap.\
-   **Spacing:** p-8, p-6, mb-4, mb-6, mt-1, mt-2, py-2, px-4.\
-   **Borders & Backgrounds:** bg-gray-100, bg-white, rounded-lg, shadow-md, border, border-gray-300, bg-gray-50, bg-gray-200.\
-   **Typography:** font-sans, text-2xl, text-xl, text-lg, text-md, text-sm, font-bold, font-semibold, font-mono, text-center, text-left, text-blue-700, text-gray-800, text-gray-600, text-gray-500, text-green-600, text-purple-700, italic.\
-   **Buttons/Inputs:** bg-blue-500, hover:bg-blue-700, text-white, rounded, transition, duration-150, ease-in-out, cursor-pointer, appearance-none.\
-   **Custom Style:** A small <style> block is included for the .slot class used in visualization, demonstrating how to add custom CSS alongside Tailwind if needed.

These classes provide a clean, responsive starting point for the simulator's appearance without writing extensive custom CSS.

## **4. Building the Simulator: Part 2 - The Brains (JavaScript & DBO Logic)**

Now, let's implement the core DBO logic in JavaScript. Create a file named script.js.

### **4.1 Representing Schedules in Code**

First, define constants and helper functions for our scheduling problem.

JavaScript

// script.js

const NUM_PATIENTS = 5;\
const NUM_SLOTS = 10;\
const MAX_EVALUATIONS = 50; // Limit the number of iterations for the demo

// --- Configuration ---\
let currentIteration = 0;\
let observedData = new Map(); // Stores evaluated schedules (key: stringified array) and their {score, iteration, acqScore, predMean, uncertainty}\
let bestScoreFound = -Infinity;\
let bestScheduleFound = null;\
let explorationFactorBeta = 1.0; // Initial Beta for UCB

// --- DOM Elements ---\
const runIterationBtn = document.getElementById('run-iteration-btn');\
const bestScoreEl = document.getElementById('best-score');\
const bestScheduleEl = document.getElementById('best-schedule');\
const historyTableBody = document.getElementById('history-table-body');\
const scheduleVisualizationEl = document.getElementById('schedule-visualization');\
const betaSlider = document.getElementById('beta-slider');\
const betaValueEl = document.getElementById('beta-value');

// --- Helper Functions ---

// Generate a random valid schedule (potential candidate)\
function generateRandomSchedule() {\
let schedule = Array(NUM_PATIENTS).fill(-1);\
let assignedSlots = new Set();\
for (let i = 0; i < NUM_PATIENTS; i++) {\
let availableSlots =;\
for (let s = 0; s < NUM_SLOTS; s++) {\
if (!assignedSlots.has(s)) {\
availableSlots.push(s);\
}\
}\
if (availableSlots.length === 0) break; // No more slots

```         
    // Decide whether to schedule this patient (e.g., 80% chance)  
    if (Math.random() < 0.9 |  
```

| i < 2) { // Ensure at least first few are scheduled const randomSlotIndex = Math.floor(Math.random() \* availableSlots.length); const assignedSlot = availableSlots; schedule\[i\] = assignedSlot; assignedSlots.add(assignedSlot); } else { schedule\[i\] = -1; // Unscheduled } } return schedule;

}

// Check if a schedule is valid (no slot conflicts)\
function isValidSchedule(schedule) {\
const assignedSlots = new Set();\
for (const slot of schedule) {\
if (slot!== -1) {\
if (assignedSlots.has(slot)) {\
return false; // Conflict\
}\
assignedSlots.add(slot);\
}\
}\
return true;\
}

// Convert schedule array to a string key for Map storage\
function scheduleToString(schedule) {\
return JSON.stringify(schedule);\
}

// Calculate distance between two schedules (simple Hamming distance variant)\
function scheduleDistance(sched1, sched2) {\
let distance = 0;\
for (let i = 0; i < NUM_PATIENTS; i++) {\
if (sched1\[i\]!== sched2\[i\]) {\
distance++;\
}\
}\
return distance;\
}

// Update the visualization of the best schedule\
function updateVisualization(schedule) {\
scheduleVisualizationEl.inner = ''; // Clear previous\
if (!schedule) {\
scheduleVisualizationEl.inner = '<span class="text-gray-500 italic">No schedule evaluated yet.</span>';\
return;\
}\
const slots = Array(NUM_SLOTS).fill(null);\
schedule.forEach((slot, patientIndex) => {\
if (slot!== -1) {\
slots\[slot\] = \`P\${patientIndex + 1}\`;\
}\
});

```         
slots.forEach((patient, index) \=> {  
    const slotEl \= document.createElement('div');  
    slotEl.classList.add('slot', 'bg-white');  
    slotEl.textContent \= \`Slot ${index}: ${patient? patient : 'Empty'}\`;  
    if (patient) {  
        slotEl.classList.add('bg-blue-100', 'font-semibold');  
    }  
    scheduleVisualizationEl.appendChild(slotEl);  
});  
```

}

// Update Beta value display\
betaSlider.oninput = function() {\
explorationFactorBeta = parseFloat(this.value);\
betaValueEl.textContent = explorationFactorBeta.toFixed(1);\
}

This sets up the basic variables, DOM references, and utility functions for creating, validating, and comparing schedules.

### **4.2 The Objective Function: Evaluating Schedule Quality (Our "Black Box")**

Now, implement the function that DBO will try to optimize. Remember, in a real application, this could be a complex simulation or external system call. Here, it's a simple calculation based on our defined objective.1

JavaScript

// (Continuing script.js)

// Objective Function: Calculate the quality score of a schedule\
// Higher score is better. Penalize later slots and unscheduled patients.\
function calculateScheduleScore(schedule) {\
if (!isValidSchedule(schedule)) {\
return -Infinity; // Invalid schedules have the worst possible score\
}

```         
let score \= 0;  
const maxPossibleSlotScore \= NUM\_SLOTS \* NUM\_PATIENTS; // Max score if all patients in slot 9

for (let i \= 0; i < NUM\_PATIENTS; i++) {  
    if (schedule\[i\]\!== \-1) {  
        // Reward scheduling earlier slots (higher score for lower slot number)  
        score \+= (NUM\_SLOTS \- 1 \- schedule\[i\]);  
    } else {  
        // Penalize unscheduled patients significantly  
        score \-= NUM\_SLOTS; // Penalty equivalent to being in the worst slot  
    }  
}

// Simulate some noise or variability (optional, makes optimization more realistic)  
// score \+= (Math.random() \- 0.5) \* 0.5; // Small random noise

return score;  
```

}

This function assigns a numerical score representing the "goodness" of a given schedule according to our simple criteria.

### **4.3 Implementing a *Simplified* Surrogate Model Concept**

As discussed (Insight 2), we'll use a simplified surrogate. It stores observed data and provides basic predictions and uncertainty estimates.

JavaScript

// (Continuing script.js)

// Simplified Surrogate Model Functions

// Predict score: Use observed value if available, otherwise use average of neighbors\
function predictScore(schedule) {\
const key = scheduleToString(schedule);\
if (observedData.has(key)) {\
return observedData.get(key).score; // Should not happen for acquisition function target\
}

```         
// Simple prediction: Average score of k-nearest neighbors (e.g., k=3)  
let k \= 3;  
let neighbors \=;  
observedData.forEach((data, obsKey) \=> {  
    const obsSched \= JSON.parse(obsKey);  
    neighbors.push({ distance: scheduleDistance(schedule, obsSched), score: data.score });  
});

neighbors.sort((a, b) \=> a.distance \- b.distance);

let sumScore \= 0;  
let count \= 0;  
for (let i \= 0; i < Math.min(k, neighbors.length); i++) {  
    sumScore \+= neighbors\[i\].score;  
    count++;  
}

if (count \=== 0) {  
    return 0; // Default prediction if no data yet  
}  
return sumScore / count;  
```

}

// Estimate uncertainty: Simple proxy based on distance to nearest observed point\
function estimateUncertainty(schedule) {\
if (observedData.size === 0) {\
return 1.0; // High uncertainty initially\
}

```         
let minDistance \= Infinity;  
observedData.forEach((data, obsKey) \=> {  
    const obsSched \= JSON.parse(obsKey);  
    const dist \= scheduleDistance(schedule, obsSched);  
    if (dist < minDistance) {  
        minDistance \= dist;  
    }  
});

// Higher uncertainty for points further away from observed data  
// Normalize crudely based on max possible distance (NUM\_PATIENTS)  
const maxDist \= NUM\_PATIENTS;  
// Use a function that increases as distance increases, capped at 1  
// Example: linear increase, or sigmoid-like  
const uncertainty \= Math.min(1.0, (minDistance / maxDist) \* 2.0); // Simple linear scaling

// Alternative: consider density \- lower density \-> higher uncertainty  
// let k \= 5; // Look at k nearest neighbors  
// let avgDistToKNeighbors \=... calculate...  
// let uncertainty \= avgDistToKNeighbors / maxDist; // Higher avg distance \-> higher uncertainty

return Math.max(0.01, uncertainty); // Ensure a small minimum uncertainty  
```

}

This predictScore gives a rough guess based on nearby known points, and estimateUncertainty provides a basic measure of how far a candidate schedule is from previously evaluated ones, serving as our proxy for uncertainty. This is a significant simplification compared to formal GP or RF models 3 but allows us to implement the core DBO loop.

### **4.4 Implementing a *Simplified* Acquisition Function (UCB)**

Using our simplified surrogate functions, we implement the UCB acquisition function logic (Insight 3). We aim to *maximize* this score.

JavaScript

// (Continuing script.js)

// Simplified Acquisition Function (Upper Confidence Bound - UCB)\
function calculateAcquisitionScore(schedule) {\
const predictedMean = predictScore(schedule);\
const uncertainty = estimateUncertainty(schedule);

```         
// UCB formula: score \= mean \+ beta \* uncertainty  
// We want to maximize this score  
const score \= predictedMean \+ explorationFactorBeta \* uncertainty;  
return { score, predictedMean, uncertainty };  
```

}

This function directly applies the UCB concept: schedules with a promising predicted score (predictedMean) or high uncertainty (uncertainty) will receive a higher acquisition score, guided by the explorationFactorBeta.14

### **4.5 Coding the DBO Loop in JavaScript**

Now, tie everything together in the main function triggered by the button click.

JavaScript

// (Continuing script.js)

// Main DBO Loop Function\
function runDboIteration() {\
if (currentIteration >= MAX_EVALUATIONS) {\
alert("Maximum evaluations reached.");\
runIterationBtn.disabled = true;\
return;\
}\
if (observedData.size === 0) {\
// --- Initial Step ---\
// Evaluate one or more initial random schedules\
const initialSchedule = generateRandomSchedule();\
const initialScore = calculateScheduleScore(initialSchedule);\
const scheduleKey = scheduleToString(initialSchedule);

```         
    observedData.set(scheduleKey, {  
        score: initialScore,  
        iteration: currentIteration,  
        acqScore: NaN, // No acquisition score for initial point  
        predMean: NaN,  
        uncertainty: NaN  
     });

    if (initialScore > bestScoreFound) {  
        bestScoreFound \= initialScore;  
        bestScheduleFound \= initialSchedule;  
    }  
    logHistory(currentIteration, initialSchedule, initialScore, NaN, NaN, NaN);  
    updateBestDisplay();  
    updateVisualization(bestScheduleFound); // Visualize initial best

} else {  
    // \--- Subsequent Steps \---  
    // 1\. Generate Candidate Schedules  
    const numCandidates \= 50; // Number of random candidates to consider  
    let candidates \=;  
    for (let i \= 0; i < numCandidates; i++) {  
        candidates.push(generateRandomSchedule());  
    }

    // 2\. Score Candidates with Acquisition Function  
    let bestCandidate \= null;  
    let maxAcquisitionScore \= \-Infinity;  
    let bestCandAcqData \= {};

    candidates.forEach(cand \=> {  
        const key \= scheduleToString(cand);  
        // Only consider schedules not already evaluated  
        if (\!observedData.has(key) && isValidSchedule(cand)) {  
            const { score: acqScore, predictedMean, uncertainty } \= calculateAcquisitionScore(cand);  
            if (acqScore > maxAcquisitionScore) {  
                maxAcquisitionScore \= acqScore;  
                bestCandidate \= cand;  
                bestCandAcqData \= { acqScore, predictedMean, uncertainty };  
            }  
        }  
    });

    // 3\. Evaluate Chosen Candidate  
    if (\!bestCandidate) {  
         // If all candidates were already seen or invalid, generate a new random one  
         bestCandidate \= generateRandomSchedule();  
         while(observedData.has(scheduleToString(bestCandidate)) ||\!isValidSchedule(bestCandidate)) {  
             bestCandidate \= generateRandomSchedule();  
             // Add a safety break if needed  
         }  
         // Recalculate acquisition data for logging, though it wasn't used for selection  
         const { score: acqScore, predictedMean, uncertainty } \= calculateAcquisitionScore(bestCandidate);  
         bestCandAcqData \= { acqScore, predictedMean, uncertainty };  
         console.warn("Could not find novel candidate via acquisition, evaluating random valid schedule.");  
    }

    const objectiveScore \= calculateScheduleScore(bestCandidate);  
    const scheduleKey \= scheduleToString(bestCandidate);

    // 4\. Add to Observed Data  
     observedData.set(scheduleKey, {  
        score: objectiveScore,  
        iteration: currentIteration,  
        acqScore: bestCandAcqData.acqScore,  
        predMean: bestCandAcqData.predictedMean,  
        uncertainty: bestCandAcqData.uncertainty  
     });

    // 5\. Update Best Found  
    if (objectiveScore > bestScoreFound) {  
        bestScoreFound \= objectiveScore;  
        bestScheduleFound \= bestCandidate;  
    }

    // 6\. Log and Update Display  
    logHistory(  
        currentIteration,  
        bestCandidate,  
        objectiveScore,  
        bestCandAcqData.acqScore,  
        bestCandAcqData.predictedMean,  
        bestCandAcqData.uncertainty  
    );  
    updateBestDisplay();  
    updateVisualization(bestScheduleFound); // Update visualization if best changed  
}

currentIteration++;  
 if (currentIteration >= MAX\_EVALUATIONS) {  
    runIterationBtn.disabled \= true;  
    runIterationBtn.textContent \= "Max Evaluations Reached";  
}  
```

}

// --- UI Update Functions ---

function updateBestDisplay() {\
bestScoreEl.textContent = bestScoreFound === -Infinity? '-' : bestScoreFound.toFixed(3);\
bestScheduleEl.textContent = bestScheduleFound? scheduleToString(bestScheduleFound) : '-';\
}

function logHistory(iter, schedule, score, acqScore, predMean, uncertainty) {\
if (iter === 0 && observedData.size === 1) {\
// Clear the initial "No evaluations" row\
historyTableBody.inner = '';\
}\
const newRow = historyTableBody.insertRow();\
newRow.inner = \`\
<td class="py-2 px-4 border-b text-sm">${iter}</td>
        <td class="py-2 px-4 border-b text-sm font-mono">${scheduleToString(schedule)}</td>\
<td class="py-2 px-4 border-b text-sm font-semibold ${score \=== bestScoreFound? 'text-green-600' : ''}">${score.toFixed(3)}</td>\
<td class="py-2 px-4 border-b text-sm">${isNaN(acqScore)? '-' : acqScore.toFixed(3)}</td>
        <td class="py-2 px-4 border-b text-sm">${isNaN(predMean)? '-' : predMean.toFixed(3)}</td>\
<td class="py-2 px-4 border-b text-sm">\${isNaN(uncertainty)? '-' : uncertainty.toFixed(3)}</td>\
\`;\
}

// --- Event Listener ---\
runIterationBtn.addEventListener('click', runDboIteration);

// --- Initial State ---\
updateBestDisplay(); // Initialize display\
updateVisualization(null); // Initialize visualization\
betaValueEl.textContent = explorationFactorBeta.toFixed(1); // Initialize beta display

This code implements the DBO loop: generating candidates, scoring them with the UCB acquisition function, selecting the best, evaluating it with the objective function, storing the result, and updating the UI.

### **4.6 Visualizing the Optimization Process**

The logHistory function updates the table, showing which schedule was evaluated at each iteration, its true score, and the acquisition function details that led to its selection. The updateBestDisplay function shows the best score and schedule found so far. The updateVisualization function provides a simple visual layout of the best schedule found. This provides crucial feedback to the user on the DBO process.

### **DBO Components Summary**

The following table maps the core DBO concepts to their simplified implementation in this tutorial:

| Component | Purpose | Research Concept (Examples) | Simplified JS Implementation |
|:-----------------|:-----------------|:-----------------|:-----------------|
| Objective Function | Defines "goodness" of a schedule (black-box) | f(x) 8 | calculateScheduleScore(schedule) function |
| Surrogate Model | Approximates objective, gives prediction+uncertainty | GP 3, RF 26 | observedData map + predictScore/estimateUncertainty functions |
| Acquisition Function | Balances exploration/exploitation to pick next point | EI 28, UCB 14 | calculateAcquisitionScore(schedule) using simplified UCB logic |
| Optimization Loop | Iteratively improves solution | Algorithm 4 | runDboIteration function tying steps together |

This table highlights how the tutorial implements the essential *roles* of each DBO component using simplified JavaScript constructs suitable for learning.

## **5. Running the Tutorial & Exploring Further**

### **5.1 Complete Code**

You should now have two files:

1.  index.: Contains the  structure and Tailwind CSS setup.\
2.  script.js: Contains all the JavaScript logic for the DBO simulation.

Ensure both files are saved in the same directory.

### **5.2 How to Run the Simulator**

1.  Save the  code provided in section 3.2 as index..\
2.  Save the complete JavaScript code provided across section 4 as script.js in the same folder.\
3.  Open the index. file in your web browser (e.g., Chrome, Firefox, Edge).

You should see the simulator interface. Click the "Run One DBO Iteration" button to perform steps of the optimization. Observe how the "Best Schedule Found" updates and how the "Evaluation History" table populates. Try adjusting the "Exploration Factor (β)" slider to see how it influences the balance between exploring uncertain schedules (higher uncertainty values in the table) and exploiting promising ones (higher predicted mean values).

### **5.3 Limitations and Simplifications Made**

This tutorial provides a conceptual introduction to DBO but involves significant simplifications necessary for a 3-hour, JavaScript-based learning experience:

1.  **Simplified Scheduling Problem:** The patient/slot scenario is highly basic, ignoring many real-world complexities like varying appointment durations, resource constraints beyond single-slot occupancy, patient preferences, emergencies, etc.\
2.  **Simplified Surrogate Model:** The surrogate model (k-nearest neighbor prediction, distance-based uncertainty) is a rudimentary placeholder. Real BO typically uses more sophisticated models like Gaussian Processes or Random Forests, which provide more principled predictions and uncertainty quantification.3 Implementing these accurately in JS is complex.\
3.  **Simplified Acquisition Function:** The UCB implementation uses proxies for mean and uncertainty. Formal acquisition functions often rely on specific properties of the surrogate model's posterior distribution (e.g., Gaussian assumptions for standard EI/UCB).14\
4.  **Basic Candidate Generation:** Candidates are generated randomly. More advanced strategies might explore the neighborhood of promising solutions more systematically.\
5.  **No Hyperparameter Tuning:** Real BO involves tuning hyperparameters of the surrogate model (e.g., kernel parameters for GPs 33) and the acquisition function (like β in UCB, though we added a slider for manual exploration). This tutorial skips automated tuning.\
6.  **Scalability:** The approach of generating random candidates and evaluating the acquisition function for each becomes inefficient as the search space grows very large. More advanced DBO methods use more sophisticated techniques to navigate vast discrete spaces.1

These simplifications were made to focus on the *core iterative loop* and the *conceptual roles* of the surrogate and acquisition function within the DBO framework, making it accessible in the given context.

### **5.4 Next Steps**

This simulator serves as a starting point. To delve deeper into Bayesian Optimization and its application to more realistic problems:

1.  **Extend the Simulator:**
    -   Implement a more complex objective function (e.g., factor in different appointment types, minimize total wait time more accurately).\
    -   Add more realistic constraints (e.g., doctor availability windows, room requirements).\
    -   Experiment with different simplified acquisition function logic (e.g., mimic EI or PI).\
    -   Try more structured candidate generation instead of pure random sampling.\
    -   Add more sophisticated visualizations of the search progress.\
2.  **Explore Python Libraries:** For serious applications, use dedicated Python libraries that provide robust implementations of state-of-the-art BO techniques:
    -   **Ax/BoTorch:** A powerful, modular framework from Meta, supporting various models, acquisition functions, multi-objective optimization, and batch optimization.3\
    -   **scikit-optimize:** Provides user-friendly BO tools, including GP-based optimization.23\
    -   **GPyOpt:** Focuses on GP-based Bayesian optimization.23\
    -   **Emukit:** A flexible Python toolkit for emulation and Bayesian optimization.35\
3.  **Learn More About Models:** Study Gaussian Processes 21 and Random Forests 24 in more detail to understand their strengths and weaknesses as surrogate models.\
4.  **Investigate Advanced Topics:** Explore concepts like batch Bayesian optimization (selecting multiple points per iteration) 8, multi-objective optimization (balancing competing goals like minimizing wait time AND maximizing utilization) 16, handling high-dimensional discrete spaces 1, and incorporating different types of data (multi-fidelity optimization).

## **6. Appendix: Quick Reference**

### **6.1 Glossary of Key Terms**

-   **Bayesian Optimization (BO):** A sequential optimization strategy for expensive-to-evaluate black-box functions, using a probabilistic model to guide the search.\
-   **Discrete Bayesian Optimization (DBO):** An adaptation of BO specifically for problems where the decision variables are chosen from a discrete set.\
-   **Black-Box Function:** An objective function whose internal workings or analytical form are unknown; it can only be evaluated for given inputs.\
-   **Surrogate Model:** An inexpensive statistical model (e.g., GP, RF) used within BO to approximate the true objective function based on observed data. Provides predictions and uncertainty estimates.\
-   **Gaussian Process (GP):** A common surrogate model based on defining a probability distribution over functions, characterized by a mean and a kernel (covariance) function.\
-   **Random Forest (RF):** An ensemble machine learning model consisting of multiple decision trees, usable as a surrogate model, especially for discrete/categorical inputs.\
-   **Acquisition Function:** A function used in BO to determine the utility of evaluating the next point, balancing exploration and exploitation based on the surrogate model's output.\
-   **Expected Improvement (EI):** An acquisition function that quantifies the expected amount of improvement over the current best observed value.\
-   **Probability of Improvement (PI):** An acquisition function that quantifies the probability of achieving *any* improvement over the current best observed value.\
-   **Upper Confidence Bound (UCB):** An acquisition function that selects points based on an optimistic estimate (mean + scaled uncertainty) of their objective value.\
-   **Exploration:** Sampling in regions of the search space where the surrogate model has high uncertainty, aiming to discover new promising areas and improve model accuracy.\
-   **Exploitation:** Sampling in regions where the surrogate model predicts good objective values, aiming to refine the solution around known optima.\
-   **Objective Function:** The function being optimized (e.g., schedule quality score).\
-   **Iteration:** One cycle of the BO loop (update model, optimize acquisition, evaluate point).

### **6.2 DBO Components Table**

| Component | Purpose | Research Concept (Examples) | Simplified JS Implementation |
|:-----------------|:-----------------|:-----------------|:-----------------|
| Objective Function | Defines "goodness" of a schedule (black-box) | f(x) 8 | calculateScheduleScore(schedule) function |
| Surrogate Model | Approximates objective, gives prediction+uncertainty | GP 3, RF 26 | observedData map + predictScore/estimateUncertainty functions |
| Acquisition Function | Balances exploration/exploitation to pick next point | EI 28, UCB 14 | calculateAcquisitionScore(schedule) using simplified UCB logic |
| Optimization Loop | Iteratively improves solution | Algorithm 4 | runDboIteration function tying steps together |

#### **Works cited**

1.  A survey and benchmark of high-dimensional Bayesian optimization of discrete sequences, accessed on April 29, 2025, <https://neurips.cc/virtual/2024/poster/97688>\
2.  Bayesian optimization - Wikipedia, accessed on April 29, 2025, <https://en.wikipedia.org/wiki/Bayesian_optimization>\
3.  Bayesian Optimization \| Ax, accessed on April 29, 2025, <https://ax.dev/docs/bayesopt/>\
4.  Bayesian optimization - Martin Krasser's Blog, accessed on April 29, 2025, <http://krasserm.github.io/2018/03/21/bayesian-optimization/>\
5.  Mastering Bayesian Optimization in Data Science \| DataCamp, accessed on April 29, 2025, <https://www.datacamp.com/tutorial/mastering-bayesian-optimization-in-data-science>\
6.  Discrete Bayesian Optimization Algorithms and Applications - Webthesis - Politecnico di Torino, accessed on April 29, 2025, <https://webthesis.biblio.polito.it/16294/1/tesi.pdf>\
7.  Bayesian optimization - What is it? How to use it best? - Inside Machine Learning, accessed on April 29, 2025, <https://inside-machinelearning.com/en/bayesian-optimization/>\
8.  BATCHED BAYESIAN OPTIMIZATION IN DISCRETE DOMAINS BY MAXIMIZING THE PROBABILITY OF INCLUDING THE OPTIMUM - OpenReview, accessed on April 29, 2025, <https://openreview.net/notes/edits/attachment?id=RWn88HJuTz&name=pdf>\
9.  Sequential closed-loop Bayesian optimization as a guide for organic molecular metallophotocatalyst formulation discovery, accessed on April 29, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11321994/>\
10. Introduction to Bayesian Optimization - YouTube, accessed on April 29, 2025, <https://www.youtube.com/watch?v=rMkweLya3T8>\
11. Robustness under parameter and problem domain alterations of Bayesian optimization methods for chemical reactions - PMC - PubMed Central, accessed on April 29, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC9434872/>\
12. High-Dimensional Discrete Bayesian Optimization with Intrinsic Dimension - ResearchGate, accessed on April 29, 2025, <https://www.researchgate.net/publication/365123643_High-Dimensional_Discrete_Bayesian_Optimization_with_Intrinsic_Dimension>\
13. batched bayesian optimization with - arXiv, accessed on April 29, 2025, <http://arxiv.org/pdf/2410.06333>\
14. Bayesian Optimization Acquisition Functions, accessed on April 29, 2025, <https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf>\
15. Bayesian Optimization Algorithm - MathWorks, accessed on April 29, 2025, <https://www.mathworks.com/help/stats/bayesian-optimization-algorithm.>\
16. 32\. Bayesian Optimization - YouTube, accessed on April 29, 2025, <https://www.youtube.com/watch?v=qVEBO1Viv7k&pp=0gcJCdgAo7VqN5tD>\
17. Bayesian Optimization: expected improvement (+LCB, PI, EV) interpretation - YouTube, accessed on April 29, 2025, <https://www.youtube.com/watch?v=PrEuA8hm9mY>\
18. Bayesian Optimization - YouTube, accessed on April 29, 2025, <https://m.youtube.com/watch?v=_SC5_2vkgbA&pp=ygUUI2JheWVzaWFuZXhwbG9yYXRpb24%3D>\
19. INFORMS TutORial: Bayesian Optimization - YouTube, accessed on April 29, 2025, <https://www.youtube.com/watch?v=c4KKvyWW_Xk>\
20. Bayesian Optimization for Beginners - Emma Benjaminson, accessed on April 29, 2025, <https://sassafras13.github.io/BayesianOptimization/>\
21. Creating a Multiphysics-Driven Gaussian Process Surrogate Model - COMSOL, accessed on April 29, 2025, <https://www.comsol.com/support/learning-center/article/creating-a-multiphysics-driven-gaussian-process-surrogate-model-94941/261>\
22. Tutorial #8: Bayesian optimization - Research Blog - RBC Borealis, accessed on April 29, 2025, <https://rbcborealis.com/research-blogs/tutorial-8-bayesian-optimization/>\
23. A Comprehensive Guide to Practical Bayesian Optimization Implementation, accessed on April 29, 2025, <https://www.numberanalytics.com/blog/comprehensive-guide-practical-bayesian-optimization-implementation>\
24. Random Forest Algorithm in Machine Learning - Analytics Vidhya, accessed on April 29, 2025, <https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/>\
25. Random Forest Regression in Python \| GeeksforGeeks, accessed on April 29, 2025, <https://www.geeksforgeeks.org/random-forest-regression-in-python/>\
26. Explainable AI (XAI) Methods Part 5- Global Surrogate Models \| Towards Data Science, accessed on April 29, 2025, <https://towardsdatascience.com/explainable-ai-xai-methods-part-5-global-surrogate-models-9c228d27e13a/>\
27. Bayesian Optimization, accessed on April 29, 2025, <https://optimization.cbe.cornell.edu/index.php?title=Bayesian_Optimization>\
28. Acquisition functions • tune, accessed on April 29, 2025, <https://tune.tidymodels.org/articles/acquisition_functions.>\
29. Acquisition Functions - BoTorch, accessed on April 29, 2025, <https://botorch.org/docs/acquisition/>\
30. Benchmarking Acquisition Functions — honegumi 0.3.1 documentation, accessed on April 29, 2025, <https://honegumi.readthedocs.io/en/v0.3.1/curriculum/tutorials/benchmarking/benchmarking.>\
31. Writing a custom acquisition function \| BoTorch, accessed on April 29, 2025, <https://botorch.org/docs/tutorials/custom_acquisition/>\
32. The Upper Confidence Bound (UCB) acquisition function balances exploration and exploitation by assigning a score of - BoTorch · Bayesian Optimization in PyTorch, accessed on April 29, 2025, <https://archive.botorch.org/v/0.6.6/tutorials/custom_acquisition>\
33. Gaussian Process Surrogate \| MOOSE, accessed on April 29, 2025, <https://mooseframework.inl.gov/releases/moose/2022-06-10/modules/stochastic_tools/examples/gaussian_process_surrogate.>\
34. Gaussian Process Surrogate - MOOSE, accessed on April 29, 2025, <https://mooseframework.inl.gov/modules/stochastic_tools/examples/gaussian_process_surrogate.>\
35. emukit/notebooks/Emukit-tutorial-Bayesian-optimization-introduction.ipynb at main - GitHub, accessed on April 29, 2025, <https://github.com/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-Bayesian-optimization-introduction.ipynb>\
36. Gaussian Process · Surrogates.jl, accessed on April 29, 2025, <https://docs.sciml.ai/Surrogates/stable/abstractgps/>\
37. RandomForest · Surrogates.jl, accessed on April 29, 2025, <https://docs.sciml.ai/Surrogates/stable/randomforest/>
