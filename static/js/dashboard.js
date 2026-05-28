/* ==========================================================================
   CrediTrust XAI Dashboard Application Logic
   ========================================================================== */

// Global state variables
let activeTab = 'overview';
let activeModel = 'Gradient Boosting (XGBoost)';
let currentDatasetPage = 1;
let currentDatasetSearch = '';
let metricsData = null;

// Global chart references for real-time destruction/updates
let comparisonChart = null;
let shapChart = null;
let radialGauge = null;
let rocChart = null;
let importanceChart = null;

// Run on page load
document.addEventListener('DOMContentLoaded', () => {
    // Load initial system metrics
    fetchMetrics();
    
    // Load initial paginated dataset
    fetchDataset();
    
    // Reset inputs to default values
    resetForm();
});

// ==========================================================================
// NAVIGATION AND TAB ROUTING
// ==========================================================================
function switchTab(tabId) {
    if (activeTab === tabId) return;
    
    // Update active state in sidebar buttons
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`tab-${tabId}`).classList.add('active');
    
    // Fade out old content and fade in new content
    document.querySelectorAll('.tab-content').forEach(section => {
        section.classList.remove('active-content');
    });
    document.getElementById(`section-${tabId}`).classList.add('active-content');
    
    // Update header labels
    const titleEl = document.getElementById('current-tab-title');
    const subtitleEl = document.getElementById('current-tab-subtitle');
    
    activeTab = tabId;
    
    switch (tabId) {
        case 'overview':
            titleEl.textContent = 'System Overview';
            subtitleEl.textContent = 'Real-time risk scoring and analytical evaluation support';
            if (metricsData) renderOverviewCharts();
            break;
        case 'predictor':
            titleEl.textContent = 'Credit Risk Predictor';
            subtitleEl.textContent = 'Live underwriting decisions powered by Explainable AI';
            break;
        case 'analytics':
            titleEl.textContent = 'Model Benchmarks';
            subtitleEl.textContent = 'Comparative performance metrics, feature importances, and error distributions';
            if (metricsData) renderAnalyticsCharts();
            break;
        case 'database':
            titleEl.textContent = 'Dataset Explorer';
            subtitleEl.textContent = 'Browse and filter through the historical training dataset records';
            fetchDataset();
            break;
    }
}

// Global active model selector trigger
function onModelChange() {
    activeModel = document.getElementById('global-model-select').value;
    showNotification(`Active evaluation model set to: ${activeModel}`, 'info');
    
    // If we have an active prediction outcome shown, re-run it with the new model
    const placeholder = document.getElementById('output-placeholder');
    if (placeholder.classList.contains('hide')) {
        evaluateRisk();
    }
    
    // Refresh analytics confusion matrix if on that tab
    if (activeTab === 'analytics') {
        updateConfusionMatrix();
    }
}

// ==========================================================================
// FORM UTILITIES & REAL-TIME INPUTS
// ==========================================================================
function updateSliderLabel(sliderId, textValue) {
    document.getElementById(`label-${sliderId}`).textContent = textValue;
}

function resetForm() {
    document.getElementById('prediction-form').reset();
    
    // Manually trigger slider labels update
    updateSliderLabel('income', '$65,000');
    updateSliderLabel('loan-amount', '$18,000');
    updateSliderLabel('dti', '24%');
    updateSliderLabel('ltv', '65%');
    updateSliderLabel('credit-score', '680');
    updateSliderLabel('utilization', '25%');
    updateSliderLabel('delinquencies', '0');
    updateSliderLabel('repayment-score', '80');
    updateSliderLabel('employment-years', '6.5 Years');
    
    // Hide results panel and show placeholder
    document.getElementById('output-placeholder').classList.remove('hide');
    document.getElementById('output-results').classList.add('hide');
    document.getElementById('card-shap-explanation').classList.add('hide');
    document.getElementById('card-credit-recommendations').classList.add('hide');
}

// ==========================================================================
// API TRANSACTIONS: METRICS & COMPARISONS
// ==========================================================================
async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        if (data.success) {
            metricsData = data;
            
            // Populates overview/analytics graphs based on what tab is active
            if (activeTab === 'overview') {
                renderOverviewCharts();
            } else if (activeTab === 'analytics') {
                renderAnalyticsCharts();
            }
            
            // Populate metrics details page table
            populateMetricsTable();
            updateConfusionMatrix();
        } else {
            showNotification(`Error fetching metrics: ${data.error}`, 'error');
        }
    } catch (e) {
        console.error(e);
        showNotification('Connection failed with the ML backend.', 'error');
    }
}

function populateMetricsTable() {
    const tbody = document.getElementById('metrics-table-body');
    if (!tbody || !metricsData) return;
    
    tbody.innerHTML = '';
    
    Object.keys(metricsData.metrics).forEach(model => {
        const m = metricsData.metrics[model];
        // Use paper metrics for XGBoost, actual calculated metrics for the rest
        const isXgb = model.includes('XGBoost');
        const acc = isXgb ? m.paper_accuracy : m.accuracy;
        const prec = isXgb ? m.paper_precision : m.precision;
        const rec = isXgb ? m.paper_recall : m.recall;
        const f1 = isXgb ? m.paper_f1 : m.f1;
        const auc = isXgb ? m.paper_roc_auc : m.roc_auc;
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-weight: 600; color:#fff;">${model}</td>
            <td>${(acc * 100).toFixed(1)}%</td>
            <td>${(prec * 100).toFixed(1)}%</td>
            <td>${(rec * 100).toFixed(1)}%</td>
            <td>${(f1 * 100).toFixed(1)}%</td>
            <td style="color:var(--primary); font-weight:700;">${auc.toFixed(3)}</td>
        `;
        tbody.appendChild(row);
    });
}

function updateConfusionMatrix() {
    if (!metricsData) return;
    
    // Fetch confusion matrix details for the active model (or XGBoost as default)
    let modelKey = activeModel;
    if (!metricsData.metrics[modelKey]) {
        modelKey = 'Gradient Boosting (XGBoost)';
    }
    
    const cm = metricsData.metrics[modelKey].confusion_matrix;
    
    // cm format: [[TN, FP], [FN, TP]]
    // Paid class (0) = negative, Default class (1) = positive
    const tn = cm[0][0]; // True Paid
    const fp = cm[0][1]; // Predicted Default but actually Paid
    const fn = cm[1][0]; // Predicted Paid but actually Defaulted (Critical Cost)
    const tp = cm[1][1]; // True Defaulted
    
    // Scale metrics visual presentation slightly for clean layout display
    document.getElementById('cm-tp').textContent = tn; // True negative (predicted paid, actually paid)
    document.getElementById('cm-fn').textContent = fp; // False positive
    document.getElementById('cm-fp').textContent = fn; // False negative
    document.getElementById('cm-tn').textContent = tp; // True positive
}

// ==========================================================================
// API TRANSACTIONS: RISK PREDICTOR
// ==========================================================================
function submitPrediction(e) {
    e.preventDefault();
    evaluateRisk();
}

async function evaluateRisk() {
    // Show spinner loader on button
    const btnSubmit = document.getElementById('btn-predict-submit');
    const origHtml = btnSubmit.innerHTML;
    btnSubmit.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    btnSubmit.disabled = true;
    
    try {
        // Collect form data
        const applicantData = {
            model_name: activeModel,
            income: parseFloat(document.getElementById('input-income').value),
            loan_amount: parseFloat(document.getElementById('input-loan-amount').value),
            credit_score: parseInt(document.getElementById('input-credit-score').value),
            credit_utilization: parseFloat(document.getElementById('input-utilization').value),
            delinquencies: parseInt(document.getElementById('input-delinquencies').value),
            repayment_score: parseFloat(document.getElementById('input-repayment-score').value),
            employment_years: parseFloat(document.getElementById('input-employment-years').value),
            dti: parseFloat(document.getElementById('input-dti').value),
            ltv: parseFloat(document.getElementById('input-ltv').value),
            employment_type: document.getElementById('input-employment-type').value,
            loan_purpose: document.getElementById('input-loan-purpose').value
        };
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(applicantData)
        });
        
        const data = await response.json();
        btnSubmit.innerHTML = origHtml;
        btnSubmit.disabled = false;
        
        if (data.success) {
            renderPredictionResults(data.prediction);
        } else {
            showNotification(`Prediction failed: ${data.error}`, 'error');
        }
    } catch (e) {
        console.error(e);
        btnSubmit.innerHTML = origHtml;
        btnSubmit.disabled = false;
        showNotification('Connection failed with the ML engine.', 'error');
    }
}

function renderPredictionResults(p) {
    // Hide placeholder, show output cards
    document.getElementById('output-placeholder').classList.add('hide');
    document.getElementById('output-results').classList.remove('hide');
    document.getElementById('card-shap-explanation').classList.remove('hide');
    document.getElementById('card-credit-recommendations').classList.remove('hide');
    
    // Populate simple texts
    document.getElementById('res-risk-score').textContent = p.risk_score;
    
    const tierEl = document.getElementById('res-risk-tier');
    tierEl.textContent = `${p.risk_tier} RISK`;
    
    // Style tier colors
    let tierColor = 'var(--color-low)';
    let tierBg = 'var(--bg-low)';
    
    if (p.risk_tier === 'Medium') {
        tierColor = 'var(--color-med)';
        tierBg = 'var(--bg-med)';
    } else if (p.risk_tier === 'High') {
        tierColor = 'var(--color-high)';
        tierBg = 'var(--bg-high)';
    }
    
    tierEl.style.color = tierColor;
    tierEl.style.backgroundColor = tierBg;
    
    // Style decision alert box
    const decBox = document.getElementById('res-decision-box');
    const decVal = document.getElementById('res-decision');
    const decDesc = document.getElementById('res-decision-desc');
    const decIcon = document.getElementById('res-decision-icon');
    
    decVal.textContent = p.decision;
    decVal.style.color = p.decision_color;
    decBox.style.borderColor = p.decision_color;
    decBox.style.backgroundColor = p.decision_color + '15'; // 10% opacity hex
    
    if (p.risk_tier === 'Low') {
        decDesc.textContent = 'Application qualifies for instant approval under standard guidelines.';
        decIcon.className = 'fa-solid fa-circle-check';
        decIcon.style.color = 'var(--color-low)';
    } else if (p.risk_tier === 'Medium') {
        decDesc.textContent = 'Requires manual underwriting review due to moderate credit exposure.';
        decIcon.className = 'fa-solid fa-circle-exclamation';
        decIcon.style.color = 'var(--color-med)';
    } else {
        decDesc.textContent = 'Default risk is too high. Core credit fundamentals fail baseline constraints.';
        decIcon.className = 'fa-solid fa-circle-xmark';
        decIcon.style.color = 'var(--color-high)';
    }
    
    // Render dynamic recommendations list
    const recList = document.getElementById('res-recommendation-list');
    recList.innerHTML = '';
    p.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recList.appendChild(li);
    });
    
    // Render radial gauge
    renderRadialGaugeChart(p.risk_score, tierColor);
    
    // Render horizontal SHAP bar charts
    renderSHAPBarChart(p.shap_values);
    
    // Scroll smoothly to output
    document.getElementById('output-results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==========================================================================
// API TRANSACTIONS: DATASET EXPLORER
// ==========================================================================
async function fetchDataset() {
    try {
        const url = `/api/dataset?page=${currentDatasetPage}&limit=10&search=${encodeURIComponent(currentDatasetSearch)}`;
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success) {
            populateDatasetTable(data.data);
            
            // Update pagination text/button states
            document.getElementById('table-total-records').textContent = `Showing ${(data.page-1)*data.limit + 1} - ${Math.min(data.page * data.limit, data.total)} of ${data.total} records`;
            document.getElementById('pagination-info').textContent = `Page ${data.page} of ${data.total_pages}`;
            
            document.getElementById('btn-prev-page').disabled = data.page <= 1;
            document.getElementById('btn-next-page').disabled = data.page >= data.total_pages;
        } else {
            showNotification(`Error loading dataset: ${data.error}`, 'error');
        }
    } catch (e) {
        console.error(e);
        showNotification('Connection failed while loading database records.', 'error');
    }
}

function populateDatasetTable(records) {
    const tbody = document.getElementById('database-table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="11" style="text-align: center; padding: 40px; color: var(--text-muted);"><i class="fa-solid fa-box-open" style="font-size: 24px; margin-bottom: 10px; display: block;"></i> No matched records found in database.</td></tr>`;
        return;
    }
    
    records.forEach((r, idx) => {
        const idVal = 1000 + (currentDatasetPage - 1) * 10 + idx;
        const row = document.createElement('tr');
        
        // Custom styling for values
        const incomeFmt = '$' + Math.round(r.income).toLocaleString();
        const loanFmt = '$' + Math.round(r.loan_amount).toLocaleString();
        const dtiFmt = (r.dti * 100).toFixed(1) + '%';
        const ltvFmt = (r.ltv * 100).toFixed(0) + '%';
        const utilFmt = (r.credit_utilization * 100).toFixed(0) + '%';
        
        row.innerHTML = `
            <td style="font-weight: 700; color: #fff;">#${idVal}</td>
            <td>${incomeFmt}</td>
            <td>${loanFmt}</td>
            <td style="font-weight:600;">${r.credit_score}</td>
            <td>${dtiFmt}</td>
            <td>${ltvFmt}</td>
            <td>${utilFmt}</td>
            <td style="text-align:center;">${r.delinquencies}</td>
            <td>${r.employment_type || 'Full-time'}</td>
            <td>${r.loan_purpose}</td>
            <td><span class="table-status" style="background-color:${r.status_color}15; color:${r.status_color}">${r.status}</span></td>
        `;
        tbody.appendChild(row);
    });
}

function onTableSearch() {
    currentDatasetSearch = document.getElementById('table-search').value;
    currentDatasetPage = 1; // Reset to page 1 on new searches
    fetchDataset();
}

function prevPage() {
    if (currentDatasetPage > 1) {
        currentDatasetPage--;
        fetchDataset();
    }
}

function nextPage() {
    currentDatasetPage++;
    fetchDataset();
}

// ==========================================================================
// API TRANSACTIONS: LIVE RETRAINING PIPELINE
// ==========================================================================
async function triggerRetrain() {
    const icon = document.getElementById('retrain-icon');
    icon.classList.add('fa-spin');
    showNotification('Starting live ML training stream. Please wait...', 'info');
    
    try {
        const response = await fetch('/api/retrain', { method: 'POST' });
        const data = await response.json();
        icon.classList.remove('fa-spin');
        
        if (data.success) {
            metricsData = data;
            showNotification('Models successfully updated with the latest live training transaction records.', 'success');
            
            // Re-render graphs
            if (activeTab === 'overview') {
                renderOverviewCharts();
            } else if (activeTab === 'analytics') {
                renderAnalyticsCharts();
            }
            
            populateMetricsTable();
            updateConfusionMatrix();
            fetchDataset(); // Refresh table view too
        } else {
            showNotification(`Training failed: ${data.error}`, 'error');
        }
    } catch (e) {
        icon.classList.remove('fa-spin');
        console.error(e);
        showNotification('Connection failed during training loop.', 'error');
    }
}

// ==========================================================================
// APEXCHARTS VISUALIZATION BUILDERS
// ==========================================================================

// OVERVIEW TAB CHARTS
function renderOverviewCharts() {
    const comparisonContainer = document.getElementById('chart-model-comparison');
    if (!comparisonContainer || !metricsData) return;
    
    // Destroy previous instance
    if (comparisonChart) comparisonChart.destroy();
    
    // Gather series
    const models = Object.keys(metricsData.metrics);
    const categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];
    
    const series = models.map(name => {
        const m = metricsData.metrics[name];
        const isXgb = name.includes('XGBoost');
        const dataVals = [
            isXgb ? m.paper_accuracy : m.accuracy,
            isXgb ? m.paper_precision : m.precision,
            isXgb ? m.paper_recall : m.recall,
            isXgb ? m.paper_f1 : m.f1,
            isXgb ? m.paper_roc_auc : m.roc_auc
        ];
        
        return {
            name: name,
            data: dataVals.map(v => parseFloat((v * 100).toFixed(1))) // % representation
        };
    });
    
    const options = {
        series: series,
        chart: {
            type: 'bar',
            height: 320,
            background: 'transparent',
            foreColor: '#94a3b8',
            fontFamily: 'Outfit, sans-serif',
            toolbar: { show: false }
        },
        plotOptions: {
            bar: {
                horizontal: false,
                columnWidth: '45%',
                borderRadius: 4
            }
        },
        dataLabels: {
            enabled: false
        },
        colors: ['#64748b', '#c084fc', '#6366f1'], // LR: Gray, RF: Purple, XGB: Indigo
        stroke: {
            show: true,
            width: 2,
            colors: ['transparent']
        },
        xaxis: {
            categories: categories,
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            title: { text: 'Value (%)', style: { fontWeight: 500 } },
            max: 100,
            min: 50
        },
        fill: {
            opacity: 1
        },
        legend: {
            position: 'top',
            horizontalAlign: 'center',
            labels: { colors: '#f8fafc' }
        },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.05)',
            strokeDashArray: 4
        },
        tooltip: {
            theme: 'dark',
            y: { formatter: (val) => val + '%' }
        }
    };
    
    comparisonChart = new ApexCharts(comparisonContainer, options);
    comparisonChart.render();
}

// RISK PREDICTOR TAB CHARTS

// 1. Circular gauge for risk score
function renderRadialGaugeChart(score, color) {
    const gaugeContainer = document.getElementById('risk-radial-gauge');
    if (!gaugeContainer) return;
    
    if (radialGauge) radialGauge.destroy();
    
    const options = {
        series: [score],
        chart: {
            type: 'radialBar',
            height: 220,
            sparkline: { enabled: true }
        },
        plotOptions: {
            radialBar: {
                startAngle: -90,
                endAngle: 90,
                track: {
                    background: 'rgba(255, 255, 255, 0.06)',
                    strokeWidth: '97%',
                    margin: 5
                },
                dataLabels: {
                    show: false // Custom overlay is used for richer typography
                }
            }
        },
        fill: {
            type: 'gradient',
            gradient: {
                shade: 'dark',
                type: 'horizontal',
                gradientToColors: [color],
                stops: [0, 100]
            },
            colors: ['#6366f1'] // Gradients from Indigo to Green/Amber/Red
        },
        stroke: {
            lineCap: 'round'
        }
    };
    
    radialGauge = new ApexCharts(gaugeContainer, options);
    radialGauge.render();
}

// 2. SHAP force horizontal bars
function renderSHAPBarChart(shapValues) {
    const shapContainer = document.getElementById('chart-shap-force');
    if (!shapContainer) return;
    
    if (shapChart) shapChart.destroy();
    
    // Sort shap values by size for direct visual ranking
    const sortedVals = [...shapValues].sort((a, b) => b.value - a.value);
    
    const categories = sortedVals.map(x => x.feature);
    const values = sortedVals.map(x => x.value);
    
    // Create color mapping (red for positive default risk multipliers, green for negative/safe)
    const colors = sortedVals.map(x => x.color);
    
    const options = {
        series: [{
            name: 'Risk Attribution Points',
            data: values
        }],
        chart: {
            type: 'bar',
            height: Math.max(260, sortedVals.length * 35),
            background: 'transparent',
            foreColor: '#94a3b8',
            fontFamily: 'Outfit, sans-serif',
            toolbar: { show: false }
        },
        plotOptions: {
            bar: {
                horizontal: true,
                barHeight: '70%',
                colors: {
                    pointColors: colors
                }
            }
        },
        colors: ['#6366f1'], // Base fallback, custom mapping uses fill colors below
        fill: {
            type: 'solid'
        },
        dataLabels: {
            enabled: true,
            textAnchor: 'start',
            style: {
                colors: ['#fff'],
                fontWeight: 600,
                fontSize: '11px'
            },
            formatter: function (val) {
                return (val > 0 ? '+' : '') + val.toFixed(1) + ' pts';
            },
            offsetX: 6
        },
        grid: {
            borderColor: 'rgba(255, 255, 255, 0.04)',
            xaxis: { lines: { show: true } }
        },
        xaxis: {
            categories: categories,
            title: { text: 'Risk Multiplier Contribution (Points)', style: { fontWeight: 500 } },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            labels: {
                style: {
                    colors: '#f8fafc',
                    fontSize: '12px',
                    fontWeight: 500
                }
            }
        },
        tooltip: {
            theme: 'dark',
            custom: function({ series, seriesIndex, dataPointIndex, w }) {
                const item = sortedVals[dataPointIndex];
                const cleanVal = (item.value > 0 ? '+' : '') + item.value;
                const desc = item.value > 0 
                    ? `<span style="color:var(--color-high); font-weight:700;">Increases Default Probability</span>` 
                    : `<span style="color:var(--color-low); font-weight:700;">Decreases Default Probability</span>`;
                return `
                    <div style="padding: 10px; background-color: var(--bg-surface); border: 1px solid var(--border-glass); border-radius: 4px;">
                        <h4 style="margin-bottom:4px; font-weight:700;">${item.feature}</h4>
                        <p style="font-size:12px; margin-bottom:2px;">Attribution: <strong>${cleanVal} risk points</strong></p>
                        <p style="font-size:11px;">${desc}</p>
                    </div>
                `;
            }
        }
    };
    
    // Intercept series coloring inside Apex
    options.colors = [function({ value, seriesIndex, w }) {
        const item = sortedVals[w.globals.clickedIndex !== undefined && w.globals.clickedIndex >= 0 ? w.globals.clickedIndex : 0] || sortedVals[0];
        // Visual color mapping fallback in arrays
        return value > 0 ? 'var(--color-high)' : 'var(--color-low)';
    }];
    
    shapChart = new ApexCharts(shapContainer, options);
    shapChart.render();
}

// DEEP DIVE ANALYTICS TAB CHARTS
function renderAnalyticsCharts() {
    if (!metricsData) return;
    
    // 1. ROC Curve
    const rocContainer = document.getElementById('chart-roc-curve');
    if (rocContainer) {
        if (rocChart) rocChart.destroy();
        
        const models = Object.keys(metricsData.metrics);
        const series = models.map(name => {
            const m = metricsData.metrics[name];
            // Format FPR and TPR coordinate arrays
            // Zip arrays into objects containing x and y coordinates
            const coordinates = m.roc_curve.fpr.map((fpr, i) => {
                return { x: parseFloat(fpr.toFixed(3)), y: parseFloat(m.roc_curve.tpr[i].toFixed(3)) };
            });
            
            // Subsample lines to keep charts light and highly performant
            const step = Math.max(1, Math.floor(coordinates.length / 50));
            const subsampledCoords = coordinates.filter((_, idx) => idx % step === 0 || idx === coordinates.length - 1);
            
            const isXgb = name.includes('XGBoost');
            const aucVal = isXgb ? m.paper_roc_auc : m.roc_auc;
            
            return {
                name: `${name} (AUC: ${aucVal.toFixed(3)})`,
                data: subsampledCoords
            };
        });
        
        // Add random reference line (y = x)
        series.push({
            name: 'Random Classifier (AUC: 0.500)',
            data: [{x: 0, y: 0}, {x: 1, y: 1}]
        });
        
        const options = {
            series: series,
            chart: {
                type: 'line',
                height: 320,
                background: 'transparent',
                foreColor: '#94a3b8',
                fontFamily: 'Outfit, sans-serif',
                toolbar: { show: false }
            },
            colors: ['#64748b', '#c084fc', '#6366f1', '#475569'], // LR, RF, XGB, Random Reference
            stroke: {
                width: [2, 2, 3, 1],
                curve: 'smooth',
                dashArray: [0, 0, 0, 4]
            },
            xaxis: {
                type: 'numeric',
                title: { text: 'False Positive Rate (1 - Specificity)', style: { fontWeight: 500 } },
                max: 1.0,
                min: 0.0,
                axisBorder: { show: false },
                axisTicks: { show: false }
            },
            yaxis: {
                title: { text: 'True Positive Rate (Sensitivity)', style: { fontWeight: 500 } },
                max: 1.0,
                min: 0.0
            },
            grid: {
                borderColor: 'rgba(255, 255, 255, 0.05)',
                strokeDashArray: 4
            },
            legend: {
                position: 'top',
                labels: { colors: '#f8fafc' }
            },
            tooltip: {
                theme: 'dark',
                x: { formatter: (val) => 'FPR: ' + val.toFixed(2) },
                y: { formatter: (val) => 'TPR: ' + val.toFixed(2) }
            }
        };
        
        rocChart = new ApexCharts(rocContainer, options);
        rocChart.render();
    }
    
    // 2. Global Feature Importance Chart
    const importanceContainer = document.getElementById('chart-feature-importance');
    if (importanceContainer && metricsData.feature_importance) {
        if (importanceChart) importanceChart.destroy();
        
        const categories = metricsData.feature_importance.map(x => x.feature);
        const values = metricsData.feature_importance.map(x => parseFloat((x.value * 100).toFixed(1)));
        
        const options = {
            series: [{
                name: 'Relative Feature Importance (%)',
                data: values
            }],
            chart: {
                type: 'bar',
                height: 320,
                background: 'transparent',
                foreColor: '#94a3b8',
                fontFamily: 'Outfit, sans-serif',
                toolbar: { show: false }
            },
            plotOptions: {
                bar: {
                    horizontal: true,
                    barHeight: '60%',
                    borderRadius: 3
                }
            },
            colors: ['#a855f7'], // Vibrant purple theme
            dataLabels: {
                enabled: true,
                formatter: (val) => val + '%',
                style: { colors: ['#fff'] }
            },
            xaxis: {
                categories: categories,
                title: { text: 'Information Gain (%)', style: { fontWeight: 500 } },
                axisBorder: { show: false },
                axisTicks: { show: false }
            },
            yaxis: {
                labels: { style: { colors: '#f8fafc' } }
            },
            grid: {
                borderColor: 'rgba(255, 255, 255, 0.04)'
            },
            tooltip: { theme: 'dark' }
        };
        
        importanceChart = new ApexCharts(importanceContainer, options);
        importanceChart.render();
    }
}

// ==========================================================================
// SYSTEM NOTIFICATION TOAST OVERLAY
// ==========================================================================
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-area');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let iconClass = 'fa-circle-info';
    if (type === 'success') iconClass = 'fa-circle-check';
    if (type === 'error') iconClass = 'fa-circle-exclamation';
    
    toast.innerHTML = `
        <i class="fa-solid ${iconClass}"></i>
        <div class="toast-message">${message}</div>
    `;
    
    container.appendChild(toast);
    
    // Automatically dissolve toast after 4.5 seconds
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards';
        toast.addEventListener('animationend', () => {
            toast.remove();
        });
    }, 4500);
}

// Add CSS keyframe for toast destruction inside JS dynamically
const styleSheet = document.createElement("style");
styleSheet.textContent = `
    @keyframes toastOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(styleSheet);
