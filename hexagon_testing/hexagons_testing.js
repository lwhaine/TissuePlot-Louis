// Global state variables
let spotChartInstance = null;
let currentProcessedDf = [];


// indices of neighbouring hexagons in hexagonal coordinates, as well as weights for sumation 
//                 / \
//               / 1/6 \
//             / \     / \
//           /     \ /     \
//        / |     1 |  1    | \
//      /1/6|       |       |1/6\
//      |  / \     / \     / \  |
//      |/     \ /     \ /     \|
//      |    1  |    1  |  1    |
//      |       |       |       |
//      |\     / \     / \     /|
//      |  \ /     \ /     \ /  |
//      \1/6|    1  |    1  |1/6/
//         \|       |       | /
//           \     / \     /
//             \ / 1/6 \ / 
//               \    /
//                 \ /
let NEIGHBOURING_HEXAGON_OFFSETS_AND_WEIGHTS = [
    [0,0,1], //central spot
    [1,0,1], //immediate neighbours
    [-1,0,1],
    [0,1,1],
    [0,-1,1],
    [-1,1,1],
    [1,-1,1],
    [1,1, 1/6], //corner spots 
    [2,-1, 1/6],
    [-1,2, 1/6],
    [-2,1, 1/6],
    [-1,-1, 1/6],
    [1,-2, 1/6],
];

// --- 1. MOCK DATA (Retained as a comment, the live data is now fetched from spot_data.csv) ---
// const MOCK_CSV_DATA = `x,y
// 100.0,100.0
// 102.1,100.5
// ...`;


// --- 2. UTILITY FUNCTIONS ---

function zip(arrays) {
    return arrays[0].map(function(_,i){
        return arrays.map(function(array){return array[i]})
    });
}

/**
 * Parses a simple CSV string into an array of objects.
 */


function parseCSV(csv) {
    const [headerLine, ...dataLines] = csv.trim().split('\n');
    const headers = headerLine.split(',').map(h => h.trim());

    // Basic validation to ensure we have 'x' and 'y' columns
    if (!headers.includes('x') || !headers.includes('y')) {
        throw new Error("CSV file must contain columns labeled 'x' and 'y'.");
    }

    return dataLines.map(line => {
        const values = line.split(',').map(v => parseFloat(v.trim()));
        const row = {};
        headers.forEach((h, i) => {
            row[h] = values[i];
        });
        return row;
    }).filter(row => !isNaN(row.x) && !isNaN(row.y)); // Filter out bad rows
}

/**
 * Calculates the mean of a number array.
 */
function calculateMean(arr) {
    if (arr.length === 0) return 0;
    const sum = arr.reduce((a, b) => a + b, 0);
    return sum / arr.length;
}

/**
 * Gets unique values from an array, sorted numerically.
 */
function getUnique(arr) {
    return Array.from(new Set(arr)).sort((a, b) => a - b);
}

/**
 * Calculates the difference between adjacent elements (simulates np.diff).
 */
function diff(arr) {
    if (arr.length <= 1) return [];
    return arr.slice(1).map((val, i) => val - arr[i]);
}

/**
 * Finds the index of the minimum value in an array (simulates np.argmin).
 */
function argmin(arr) {
    let min = Infinity;
    let minIndex = -1;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] < min) {
            min = arr[i];
            minIndex = i;
        }
    }
    return minIndex;
}

/**
 * Simplified 1D DBSCAN for min_samples=1, used for grouping close points.
 * Assigns a new label when the distance between consecutive sorted points >= eps.
 */
function dbscan1D(data, eps) {
    if (data.length === 0) return [];

    // 1. Sort by value, keeping track of the original index
    const sortedDataWithIndex = data.map((value, index) => ({ index, value }))
                                   .sort((a, b) => a.value - b.value);

    const labels = new Array(data.length).fill(-1);
    let currentLabel = 0;

    for (let i = 0; i < sortedDataWithIndex.length; i++) {
        const item = sortedDataWithIndex[i];

        // Check distance to the previous item (if one exists)
        if (i === 0 || (item.value - sortedDataWithIndex[i-1].value) >= eps) {
            // Start a new cluster
            currentLabel++;
        }

        // Assign the cluster label to the original index
        labels[item.index] = currentLabel;
    }
    return labels;
}

function aggregate_spot_property(spots,weights,property_name) {
    let total_property_val = 0;
    let total_weights = 0;
    for (let i=0; i<spots.length; i++){
        // is this how you access an arbitrary object property?
        total_property_val += spots[i][property_name] * weights[i]
        total_weights += weights[i];
    }
    return total_property_val / total_weights;
}

function construct_empty_spot_arrays(n_levels,level_0_nx,level_0_ny) {
    let this_level_nx = level_0_nx;
    let this_level_ny = level_0_ny;

    let levels = new Array();

    for (let i = 0; i< n_levels; i++){
        console.log("Creating spot array for level ", i, " with dimensions ",this_level_nx,this_level_ny);
        const this_level_grid = Array(this_level_ny + 1).fill(null).map(() => Array(this_level_nx + 1).fill(null));
        levels.push(this_level_grid);

        this_level_nx = Math.ceil(this_level_nx / 3)
        this_level_ny = Math.ceil(this_level_ny / 3)

    }
    return levels;
}

// --- 3. MAIN LOGIC FUNCTION ---

/**
 * Core function to process the spot data.
 */
function processSpotData(csvData) {
    let df = parseCSV(csvData);

    if (df.length === 0) {
        throw new Error("CSV data is empty or malformed. Ensure it has 'x' and 'y' columns.");
    }

    console.log("Initial Data Rows:", df.length);

    // 1. Convert from hexagonal to square coords
    const SQRT3_DIV_3 = Math.sqrt(3) / 3;
    df.forEach(row => {
        row.hex_x = row.x - SQRT3_DIV_3 * row.y;
        row.hex_y = row.y;
    });
    const hex_x_data = df.map(row => row.hex_x);
    const hex_y_data = df.map(row => row.hex_y);

    // 2. Cluster in the x and y axes separately
    const DBSCAN_EPS = 40;
    const x_labels = dbscan1D(hex_x_data, DBSCAN_EPS);
    const y_labels = dbscan1D(hex_y_data, DBSCAN_EPS);

    df.forEach((row, i) => {
        row.x_labels_dbscan = x_labels[i];
        row.y_labels_dbscan = y_labels[i];
    });

    // 3. Group and calculate mean for each cluster (group_x, group_y)
    const groupMeansX = {};
    const groupCountsX = {};
    const groupMeansY = {};
    const groupCountsY = {};

    df.forEach(row => {
        const lx = row.x_labels_dbscan;
        const ly = row.y_labels_dbscan;

        groupMeansX[lx] = (groupMeansX[lx] || 0) + row.hex_x;
        groupCountsX[lx] = (groupCountsX[lx] || 0) + 1;

        groupMeansY[ly] = (groupMeansY[ly] || 0) + row.hex_y;
        groupCountsY[ly] = (groupCountsY[ly] || 0) + 1;
    });

    // Finalize group means and assign back
    df.forEach(row => {
        const lx = row.x_labels_dbscan;
        const ly = row.y_labels_dbscan;
        row.group_x = groupMeansX[lx] / groupCountsX[lx];
        row.group_y = groupMeansY[ly] / groupCountsY[ly];
    });

    // 4. Calculate mean grid spacing
    const unique_group_x = getUnique(df.map(r => r.group_x));
    const unique_group_y = getUnique(df.map(r => r.group_y));

    const x_dists = diff(unique_group_x);
    const y_dists = diff(unique_group_y);

    const mean_x_dist = calculateMean(x_dists);
    const mean_y_dist = calculateMean(y_dists);

    // Recalculate mean spacing, excluding points spaced further than 1.5x the mean
    const recalc_x_dists = x_dists.filter(d => Math.abs(d - mean_x_dist) < (mean_x_dist / 2));
    const recalc_y_dists = y_dists.filter(d => Math.abs(d - mean_y_dist) < (mean_y_dist / 2));

    const recalc_mean_x_dist = calculateMean(recalc_x_dists.length > 0 ? recalc_x_dists : [mean_x_dist]);
    const recalc_mean_y_dist = calculateMean(recalc_y_dists.length > 0 ? recalc_y_dists : [mean_y_dist]);

    console.log(`Recalculated Mean X Spacing: ${recalc_mean_x_dist.toFixed(2)}`);
    console.log(`Recalculated Mean Y Spacing: ${recalc_mean_y_dist.toFixed(2)}`);

    // 5. Construct grid lines
    const xmin = Math.min(...unique_group_x);
    const xmax = Math.max(...unique_group_x);
    const ymin = Math.min(...unique_group_y);
    const ymax = Math.max(...unique_group_y);

    // Determine grid size
    const n_x = Math.floor(Math.abs(xmax - xmin) / recalc_mean_x_dist) + 2;
    const n_y = Math.floor(Math.abs(ymax - ymin) / recalc_mean_y_dist) + 1;

    const gridlines_x = Array.from({ length: n_x }, (_, i) => xmin + i * recalc_mean_x_dist);
    const gridlines_y = Array.from({ length: n_y }, (_, i) => ymin + i * recalc_mean_y_dist);

    // 6. Assign each spot to a gridline index
    let max_grid_x = 0;
    let max_grid_y = 0;

    df.forEach(row => {
        const dists_x = gridlines_x.map(gx => Math.abs(gx - row.group_x));
        const dists_y = gridlines_y.map(gy => Math.abs(gy - row.group_y));

        const x_grid_ix = argmin(dists_x);
        const y_grid_ix = argmin(dists_y);

        row.grid_x = x_grid_ix;
        row.grid_y = y_grid_ix;

        max_grid_x = Math.max(max_grid_x, x_grid_ix);
        max_grid_y = Math.max(max_grid_y, y_grid_ix);
    });

    // because max_grid_x is an array index and starts counting from zero
    let n_gridlines_x = max_grid_x + 1;
    let n_gridlines_y = max_grid_y + 1;
    console.log("Max Grid Indices:", n_gridlines_x, n_gridlines_y);

    const spots = construct_empty_spot_arrays(3,n_gridlines_x,n_gridlines_y);

    for (let i=0;i<n_gridlines_x;i++){
        for (let j=0;j<n_gridlines_y;j++){
            const spot = {
                x: gridlines_x[i] - SQRT3_DIV_3 * gridlines_y[i],
                y: gridlines_y[j],
                radius: 28.477165 + 40, //This shouldn't be hardcoded BUT IT IS FOR NOW
                empty: true,
                gene_expr:0,
            };
            spots[0][i][j] = spot
        }
    }
    
    df.forEach(row => {
        const spot = {
            x: row.x,
            y: row.y,
            hex_x: row.hex_x,
            hex_y: row.hex_y,
            grid_x: row.grid_x,
            grid_y: row.grid_y,
            radius: row.radius + 40,
            gene_expr:1,
            empty: false,
        };

        const y_index = row.grid_y;
        const x_index = row.grid_x;

        if (spots[0][y_index] && spots[0][y_index][x_index] === null) {
            spots[0][y_index][x_index] = [];
        }

        if (spots[0][y_index] && spots[0][y_index][x_index]) {
            spots[0][y_index][x_index] = spot;
        }
    });

    let previous_level = spots[0];

    for (this_level=1;this_level<spots.length;this_level++){
        console.log('aggregating level', this_level)
        for (i=0;i<previous_level.length;i++){
            for (j=0;previous_level[0].length;j++){
                if (i%3==0 & j%3 == 0){
                    let spot = previous_level[i][j];

                    let spots_to_aggregate = new Array();
                    let weights_to_aggregate = new Array();

                    for(let hex of NEIGHBOURING_HEXAGON_OFFSETS_AND_WEIGHTS){{
                        let offset_i = hex[0] + i;
                        let offset_j = hex[1] + j;
                        let weight = hex[2];

                        if (offset_i < 0 || offset_i >= previous_level.length){
                            continue;    
                        }
                        if (offset_j < 0 || offset_j >= previous_level[0].length){
                            continue;
                        }

                        let candidate = previous_level[offset_i][offset_j];

                        if (!candidate.empty){
                            spots_to_aggregate.push(candidate);
                            weights_to_aggregate.push(weight);
                        }

                    }

                    let is_aggregated_spot_empty = (spots_to_aggregate.length);

                    // aggregate spot properties here, needs updating
                    let total_expr = null
                    if (!is_aggregated_spot_empty){
                        total_expr = aggregate_spot_property(spots_to_aggregate,weights_to_aggregate,'gene_expr')
                    }

                    let aggregated_spot = {
                        x: spot.x,
                        y: spot.y,
                        hex_x: spot.hex_x,
                        hex_y: spot.hex_y,
                        grid_x: spot.grid_x,
                        grid_y: spot.grid_y,
                        gene_expr:total_expr,
                        radius: 3 * spot.radius,
                        empty: is_aggregated_spot_empty,
                    }
                    spots[this_level][Math.floor(i/3)][Math.floor(j/3)] = aggregated_spot
                        }
                }
            }
        }
        previous_level = spots[this_level];
    }
                

    return { spots, df };
}

// --- 4. PLOTTING LOGIC (Chart.js) ---

/** Generates a distinct color for each cluster index. */
function generateColor(labelIndex) {
    const colors = [
        '#D32F2F', '#1976D2', '#388E3C', '#FBC02D', '#7B1FA2',
        '#0097A7', '#E64A19', '#5D4037', '#689F38', '#C2185B'
    ];
    // Uses a simple modulo to cycle through a set of distinct colors
    return colors[labelIndex % colors.length];
}

/** Renders the scatter plot using the processed data and selected color axis. */
function renderPlot(df, colorBy, canvasId) {
    const ctx = document.getElementById(canvasId);
    if (!df || df.length === 0) return;

    // Group data by the selected label (grid_x or grid_y)
    const groupedData = df.reduce((acc, row) => {
        const label = row[colorBy];
        if (!acc[label]) {
            acc[label] = {
                label: `${colorBy.toUpperCase()} ${label}`,
                data: [],
                backgroundColor: generateColor(label),
                pointRadius: 5,
                borderColor: generateColor(label),
                borderWidth: 1,
            };
        }
        acc[label].data.push({ x: row.x, y: row.y });
        return acc;
    }, {});
    
    const datasets = Object.values(groupedData);

    if (spotChartInstance) {
        spotChartInstance.destroy(); // Destroy previous chart instance
    }

    spotChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            aspectRatio: 2,
            plugins: {
                title: {
                    display: true,
                    text: `Spot Positions Colored by ${colorBy.toUpperCase()}`
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        boxHeight: 8,
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const point = context.parsed;
                            const datasetIndex = context.datasetIndex;
                            const labelKey = Object.keys(groupedData)[datasetIndex];
                            
                            // Find the original data point from the df array based on the dataset
                            const originalData = df.find(r => r[colorBy] === parseInt(labelKey) && r.x === point.x && r.y === point.y);
                            if (originalData) {
                                return `(X: ${point.x.toFixed(1)}, Y: ${point.y.toFixed(1)}) Grid: (${originalData.grid_x}, ${originalData.grid_y})`;
                            }
                            return `(X: ${point.x.toFixed(1)}, Y: ${point.y.toFixed(1)})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Original X Coordinate',
                        font: { weight: 'bold' }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Original Y Coordinate',
                        font: { weight: 'bold' }
                    }
                }
            }
        }
    });
}


// --- 5. EXECUTION & DATA FETCHING ---

/**
 * Updates the UI with the processed data.
 */
function updateUI(finalGrid, df) {
     currentProcessedDf = df;

    // 1. Update JSON Output
    const outputDiv = document.getElementById('output');
    const jsonString = JSON.stringify(finalGrid, (key, value) => {
        if (typeof value === 'number' && key !== 'grid_x' && key !== 'grid_y') {
            return parseFloat(value.toFixed(2));
        }
        return value;
    }, 2);
    outputDiv.textContent = jsonString;

    // 2. Update Plot
    // const colorBySelect = document.getElementById('colorBy');
    // renderPlot(df, colorBySelect.value, 'spotChart');

    // 3. Re-enable the plot change listener with the new data
    // colorBySelect.onchange = (e) => {
    //     renderPlot(currentProcessedDf, e.target.value, 'spotChart');
    // };
}

/**
 * Handles error reporting in the UI.
 */
function reportError(message) {
    document.getElementById('output').textContent = `Error: ${message}`;
    console.error(message);
    if (spotChartInstance) {
        spotChartInstance.destroy();
        spotChartInstance = null;
    }
    currentProcessedDf = [];
}

/**
 * Main function to fetch and process data on load.
 */
async function processAndDisplayData() {
    document.getElementById('output').textContent = "Attempting to fetch and process 'spot_data.csv'...";

    // try {
    {
        // Fetch the file, assuming it's in the same directory
        const response = await fetch('SpotPositions_SP2.csv');

        if (!response.ok) {
            throw new Error(`Failed to fetch 'spot_data.csv'. Status: ${response.status}`);
        }

        const csvData = await response.text();
        
        // Process the data
        const { spots, df } = processSpotData(csvData);
        // updateUI(finalGrid, df);
        console.log("Data processing complete.");
        console.log("Final 2D Grid Array (Row: grid_y, Column: grid_x):", spots);

    } 

    // catch (error) {
    //     reportError(error.message);
    // }
}

// Kick off the process on window load
window.onload = processAndDisplayData;
