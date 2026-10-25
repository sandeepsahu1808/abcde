document.addEventListener('DOMContentLoaded', async () => {
    const typeSelect = document.getElementById('equipment-type');
    const dataFileLabel = document.getElementById('data-file-label');
    const form = document.getElementById('upload-form');
    const btn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const resultsDiv = document.getElementById('results');
    const emptyState = document.getElementById('empty-state');
    const summaryDiv = document.getElementById('summary');
    const tbody = document.querySelector('#results-table tbody');
    const loader = document.getElementById('loader');

    const outcomeDetailsDiv = document.getElementById('outcome-details');
    const pageSizeSelect = document.getElementById('page-size');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const pageNumbersContainer = document.getElementById('page-numbers');
    const downloadExcelBtn = document.getElementById('download-excel-btn');
    const exportBtnText = document.getElementById('export-btn-text');

    let currentData = [];
    let filteredData = [];
    let currentNullImputations = [];
    let currentPage = 1;
    let pageSize = 10;
    let currentFileName = 'telemetry_data';

    // Fetch available types from backend
    try {
        const res = await fetch('/health');
        const data = await res.json();
        data.equipment_types.forEach(type => {
            const opt = document.createElement('option');
            opt.value = type;
            if (type === 'static_rotating') {
                opt.textContent = "Static machine";
            } else if (type === 'cyclic_sts') {
                opt.textContent = "Cyclic machine";
            } else {
                opt.textContent = type;
            }
            typeSelect.appendChild(opt);
        });
        
        // Path-based routing: detect if we hit /cyclic_sts or /static_rotating
        const path = window.location.pathname.substring(1);
        if (data.equipment_types.includes(path)) {
            typeSelect.value = path;
        }
        updateFormState();

    } catch (e) {
        console.error("Failed to fetch health check", e);
    }

    typeSelect.addEventListener('change', () => {
        updateFormState();
        // Update URL based on UI routing
        history.pushState(null, '', `/${typeSelect.value}`);
    });

    function updateFormState() {
        if (dataFileLabel) {
            if (typeSelect.value && typeSelect.value.startsWith('cyclic')) {
                dataFileLabel.textContent = "1Hz SCADA Telemetry (Excel/CSV)";
            } else {
                dataFileLabel.textContent = "Telemetry Data (Excel/CSV)";
            }
        }
    }


    pageSizeSelect.addEventListener('change', (e) => {
        pageSize = parseInt(e.target.value);
        currentPage = 1;
        renderTablePage();
    });

    prevBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderTablePage();
        }
    });

    nextBtn.addEventListener('click', () => {
        if (currentPage * pageSize < filteredData.length) {
            currentPage++;
            renderTablePage();
        }
    });

    const filterParameter = document.getElementById('filter-parameter');
    const filterFault = document.getElementById('filter-fault');
    const filterOutcome = document.getElementById('filter-outcome');

    function applyFilters() {
        filteredData = currentData.filter(r => {
            const p = filterParameter.value === 'All' || r['Parameter Category'] === filterParameter.value;
            const f = filterFault.value === 'All' || (r['Anomaly Fault'] || 'Unknown') === filterFault.value;
            const o = filterOutcome.value === 'All' || r['Detection Outcome'] === filterOutcome.value;
            return p && f && o;
        });
        currentPage = 1;
        renderTablePage();
    }

    if (filterParameter) filterParameter.addEventListener('change', applyFilters);
    if (filterFault) filterFault.addEventListener('change', applyFilters);
    if (filterOutcome) filterOutcome.addEventListener('change', applyFilters);

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        btn.disabled = true;
        btnText.textContent = "Processing...";
        loader.style.display = 'block';
        resultsDiv.style.display = 'none';
        emptyState.style.display = 'flex';

        const fileInput = document.getElementById('data-file');
        if (fileInput.files.length > 0) {
            currentFileName = fileInput.files[0].name;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);


        try {
            const response = await fetch(`/detect?equipment_type=${typeSelect.value}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let errMsg = response.statusText;
                try {
                    const err = await response.json();
                    errMsg = err.detail || errMsg;
                } catch (_) {
                    try {
                        const txt = await response.text();
                        if (txt) errMsg = txt;
                    } catch (_) {}
                }
                alert("Server Error (" + response.status + "): " + errMsg);
                return;
            }

            const data = await response.json();
            renderResults(data);
            
        } catch (error) {
            alert("Network error: " + error.message);
        } finally {
            btn.disabled = false;
            btnText.textContent = "Run Analysis";
            loader.style.display = 'none';
        }
    });

    function renderResults(data) {
        emptyState.style.display = 'none';
        currentData = data.results;
        currentNullImputations = data.null_imputations || [];
        currentPage = 1;

        summaryDiv.innerHTML = `
            <div class="card"><h3>Rows Scored</h3><p>${data.rows_scored.toLocaleString()}</p></div>
            <div class="card"><h3>Total Episodes</h3><p>${data.anomalies_found.toLocaleString()}</p></div>
        `;
        
        // Show detail breakdown instead of KPI boxes
        let detailsHtml = '';
        if (data.summary) {
            for (const [key, value] of Object.entries(data.summary)) {
                detailsHtml += `<span><strong>${key}:</strong> ${value}</span>`;
            }
        }
        outcomeDetailsDiv.innerHTML = detailsHtml;

        // Render dynamic telemetry graph for static sensors
        initTelemetryGraph(data.sensor_timeseries);

        function getCounts(field) {
            const counts = {};
            currentData.forEach(r => {
                let val = r[field];
                if (field === 'Anomaly Fault' && !val) val = 'Unknown';
                if (!val) return;
                counts[val] = (counts[val] || 0) + 1;
            });
            return counts;
        }

        function populateSelect(selectElem, counts) {
            if (!selectElem) return;
            selectElem.innerHTML = '<option value="All">(Select All)</option>';
            Object.keys(counts).sort().forEach(item => {
                const opt = document.createElement('option');
                opt.value = item;
                opt.textContent = `${item} (${counts[item]})`;
                selectElem.appendChild(opt);
            });
        }
        
        populateSelect(filterParameter, getCounts('Parameter Category'));
        populateSelect(filterFault, getCounts('Anomaly Fault'));
        populateSelect(filterOutcome, getCounts('Detection Outcome'));

        const isCyclic = (data.equipment_type && data.equipment_type.startsWith('cyclic')) ||
                         (currentData.length > 0 && ('Cycle ID' in currentData[0] || 'cycle_id' in currentData[0])) ||
                         (typeSelect && typeSelect.value.startsWith('cyclic'));
        const thead = document.getElementById('table-head');
        if (thead) {
            if (isCyclic) {
                thead.innerHTML = `
                    <tr>
                        <th>Asset ID</th>
                        <th>Cycle ID</th>
                        <th>Phase</th>
                        <th>Parameter</th>
                        <th>Detected Value</th>
                        <th>Anomaly Fault</th>
                        <th>Data Confidence</th>
                        <th>Start Time</th>
                        <th>Outcome Status</th>
                    </tr>
                `;
            } else {
                thead.innerHTML = `
                    <tr>
                        <th>Asset ID</th>
                        <th>Parameter</th>
                        <th>Detected Value</th>
                        <th>Anomaly Fault</th>
                        <th>Data Confidence</th>
                        <th>Start Time</th>
                        <th>Outcome Status</th>
                    </tr>
                `;
            }
        }

        resultsDiv.style.display = 'block';
        applyFilters();
    }

    function renderTablePage() {
        tbody.innerHTML = '';
        
        const totalRows = filteredData.length;
        const totalPages = Math.ceil(totalRows / pageSize) || 1;
        
        const start = (currentPage - 1) * pageSize;
        const end = Math.min(start + pageSize, totalRows);
        
        const pageData = filteredData.slice(start, end);
        const isCyclic = (currentData.length > 0 && ('Cycle ID' in currentData[0] || 'cycle_id' in currentData[0])) ||
                         (typeSelect && typeSelect.value.startsWith('cyclic'));
        
        pageData.forEach(r => {
            const tr = document.createElement('tr');
            
            let badgeClass = 'Suspect';
            if (r['Detection Outcome'] === 'Anomaly') badgeClass = 'Anomaly';
            if (r['Detection Outcome'] === 'Watch') badgeClass = 'Watch';

            const rawConf = r['Data Confidence'] !== undefined ? r['Data Confidence'] : (r['confidence'] !== undefined ? r['confidence'] : '95.0%');
            const confStr = typeof rawConf === 'number' ? `${rawConf.toFixed(1)}%` : String(rawConf);

            if (isCyclic) {
                const cycleVal = r['Cycle ID'] !== undefined && r['Cycle ID'] !== null && r['Cycle ID'] !== '' ? r['Cycle ID'] : (r['cycle_id'] || '-');
                const phaseVal = r['Phase'] !== undefined && r['Phase'] !== null && r['Phase'] !== '' ? r['Phase'] : (r['phase'] || '-');
                tr.innerHTML = `
                    <td><strong>${r['Asset Category']}</strong> <span style="color:var(--text-muted);font-size:0.8rem">(${r['Object ID']})</span></td>
                    <td><span class="cycle-tag">${cycleVal}</span></td>
                    <td><span class="phase-tag">${phaseVal}</span></td>
                    <td>${r['Parameter Category']}</td>
                    <td>${r['Detected Value']}</td>
                    <td>${r['Anomaly Fault'] || 'Unknown'}</td>
                    <td><span class="conf-badge">${confStr}</span></td>
                    <td>${r['Anomaly Start Date/Time']}</td>
                    <td><span class="badge ${badgeClass}">${r['Detection Outcome']}</span></td>
                `;
            } else {
                tr.innerHTML = `
                    <td><strong>${r['Asset Category']}</strong> <span style="color:var(--text-muted);font-size:0.8rem">(${r['Object ID']})</span></td>
                    <td>${r['Parameter Category']}</td>
                    <td>${r['Detected Value']}</td>
                    <td>${r['Anomaly Fault'] || 'Unknown'}</td>
                    <td><span class="conf-badge">${confStr}</span></td>
                    <td>${r['Anomaly Start Date/Time']}</td>
                    <td><span class="badge ${badgeClass}">${r['Detection Outcome']}</span></td>
                `;
            }
            tbody.appendChild(tr);
        });
        
        renderPageNumbers(totalPages);
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === totalPages || totalPages === 0;
    }

    function renderPageNumbers(totalPages) {
        if (!pageNumbersContainer) return;
        pageNumbersContainer.innerHTML = '';
        
        let start = Math.max(1, currentPage - 2);
        let end = Math.min(totalPages, currentPage + 2);
        
        if (currentPage <= 3) end = Math.min(totalPages, 5);
        if (currentPage >= totalPages - 2) start = Math.max(1, totalPages - 4);
        
        for (let i = start; i <= end; i++) {
            const btn = document.createElement('button');
            btn.className = 'page-num-btn' + (i === currentPage ? ' active' : '');
            btn.textContent = i;
            btn.addEventListener('click', () => {
                currentPage = i;
                renderTablePage();
            });
            pageNumbersContainer.appendChild(btn);
        }
    }

    const fileInputElem = document.getElementById('data-file');
    if (fileInputElem) {
        fileInputElem.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                currentFileName = e.target.files[0].name;
            }
        });
    }

    if (downloadExcelBtn) {
        downloadExcelBtn.addEventListener('click', async () => {
            if (!currentData || currentData.length === 0) {
                alert("No anomaly incidents found to export.");
                return;
            }

            const originalText = exportBtnText ? exportBtnText.textContent : "Download Excel";
            downloadExcelBtn.disabled = true;
            downloadExcelBtn.classList.add('loading');
            if (exportBtnText) exportBtnText.textContent = "Exporting...";

            try {
                const response = await fetch('/export-excel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        records: currentData,
                        filtered_records: filteredData,
                        equipment_type: typeSelect.value,
                        filename: currentFileName,
                        null_imputations: currentNullImputations
                    })
                });

                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}: ${response.statusText}`);
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;

                let downloadName = `${currentFileName.replace(/\.[^/.]+$/, "")}_Anomalies.xlsx`;
                const disposition = response.headers.get('Content-Disposition');
                if (disposition && disposition.indexOf('filename=') !== -1) {
                    const matches = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/.exec(disposition);
                    if (matches != null && matches[1]) {
                        downloadName = matches[1].replace(/['"]/g, '');
                    }
                }
                a.download = downloadName;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);

                if (exportBtnText) exportBtnText.textContent = "Downloaded!";
                downloadExcelBtn.classList.remove('loading');
                downloadExcelBtn.classList.add('success');
                setTimeout(() => {
                    if (exportBtnText) exportBtnText.textContent = originalText;
                    downloadExcelBtn.disabled = false;
                    downloadExcelBtn.classList.remove('success');
                }, 1800);

            } catch (err) {
                console.warn("Server Excel export error, using CSV fallback:", err);
                fallbackCsvDownload(filteredData.length > 0 ? filteredData : currentData, currentFileName);
                if (exportBtnText) exportBtnText.textContent = "Downloaded (CSV)!";
                downloadExcelBtn.classList.remove('loading');
                setTimeout(() => {
                    if (exportBtnText) exportBtnText.textContent = originalText;
                    downloadExcelBtn.disabled = false;
                }, 1800);
            }
        });
    }

    function fallbackCsvDownload(data, fileName) {
        if (!data || !data.length) return;
        const headers = Object.keys(data[0]);
        const csvRows = [headers.join(',')];
        data.forEach(row => {
            const values = headers.map(h => {
                const val = (row[h] === null || row[h] === undefined) ? '' : String(row[h]);
                return `"${val.replace(/"/g, '""')}"`;
            });
            csvRows.push(values.join(','));
        });
        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        const base = (fileName || 'telemetry_data').replace(/\.[^/.]+$/, "");
        a.download = `${base}_Anomalies.csv`;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    // ==========================================
    // Dynamic Sensor Telemetry & Anomaly Outlier Graph
    // ==========================================
    let telemetryChartInstance = null;
    let currentSensorTimeseries = null;

    const graphSection = document.getElementById('graph-section');
    const graphSensorSelect = document.getElementById('graph-sensor-select');
    const graphAssetSelect = document.getElementById('graph-asset-select');
    const graphAssetGroup = document.getElementById('graph-asset-group');
    const graphStatHealthy = document.getElementById('graph-stat-healthy');
    const graphStatAnomaly = document.getElementById('graph-stat-anomaly');
    const graphStatSuspect = document.getElementById('graph-stat-suspect');
    const graphStatRate = document.getElementById('graph-stat-rate');
    const graphStatLimitPill = document.getElementById('graph-stat-limit-pill');
    const graphStatLimit = document.getElementById('graph-stat-limit');
    const telemetryCanvas = document.getElementById('telemetry-chart');

    if (graphSensorSelect) {
        graphSensorSelect.addEventListener('change', () => updateChartDisplay());
    }
    if (graphAssetSelect) {
        graphAssetSelect.addEventListener('change', () => updateChartDisplay());
    }

    function initTelemetryGraph(timeseriesData) {
        if (!timeseriesData || !timeseriesData.sensors || Object.keys(timeseriesData.sensors).length === 0) {
            if (graphSection) graphSection.style.display = 'none';
            if (telemetryChartInstance) {
                telemetryChartInstance.destroy();
                telemetryChartInstance = null;
            }
            currentSensorTimeseries = null;
            return;
        }

        currentSensorTimeseries = timeseriesData;
        if (graphSection) graphSection.style.display = 'block';

        // Populate Sensor Channel Dropdown
        if (graphSensorSelect) {
            graphSensorSelect.innerHTML = '';
            for (const [colName, info] of Object.entries(timeseriesData.sensors)) {
                const opt = document.createElement('option');
                opt.value = colName;
                opt.textContent = `${info.name} (${info.unit})`;
                graphSensorSelect.appendChild(opt);
            }
            if (timeseriesData.default_sensor && timeseriesData.sensors[timeseriesData.default_sensor]) {
                graphSensorSelect.value = timeseriesData.default_sensor;
            }
        }

        // Populate Asset ID Dropdown
        if (graphAssetSelect) {
            graphAssetSelect.innerHTML = '<option value="All">(Select All)</option>';
            if (timeseriesData.assets && timeseriesData.assets.length > 0) {
                timeseriesData.assets.forEach(asset => {
                    const opt = document.createElement('option');
                    opt.value = asset;
                    opt.textContent = `Asset: ${asset}`;
                    graphAssetSelect.appendChild(opt);
                });
            }
            graphAssetSelect.value = 'All';
        }

        updateChartDisplay();
    }

    function updateChartDisplay() {
        if (!currentSensorTimeseries || !currentSensorTimeseries.sensors || !telemetryCanvas) return;

        const sensorKey = graphSensorSelect ? graphSensorSelect.value : null;
        if (!sensorKey || !currentSensorTimeseries.sensors[sensorKey]) return;

        const sensorInfo = currentSensorTimeseries.sensors[sensorKey];
        const selectedAsset = graphAssetSelect ? graphAssetSelect.value : 'All';

        let points = sensorInfo.points || [];
        if (selectedAsset !== 'All') {
            points = points.filter(p => p.asset === selectedAsset);
        } else {
            // Sort by timestamp when viewing all assets
            points = [...points].sort((a, b) => (a.t > b.t ? 1 : (a.t < b.t ? -1 : 0)));
        }

        // Calculate KPI Badges
        const totalCount = points.length;
        const anomalyCount = points.filter(p => p.status === 'anomaly').length;
        const suspectCount = points.filter(p => p.status === 'suspect').length;
        const healthyCount = points.filter(p => p.status === 'healthy').length;
        const outlierRate = totalCount > 0 ? (((anomalyCount + suspectCount) / totalCount) * 100).toFixed(1) + '%' : '0.0%';

        if (graphStatHealthy) graphStatHealthy.textContent = healthyCount.toLocaleString();
        if (graphStatAnomaly) graphStatAnomaly.textContent = anomalyCount.toLocaleString();
        if (graphStatSuspect) graphStatSuspect.textContent = suspectCount.toLocaleString();
        if (graphStatRate) graphStatRate.textContent = outlierRate;

        // Threshold pill text
        if (graphStatLimitPill && graphStatLimit) {
            if (sensorInfo.limit_min !== null && sensorInfo.limit_max !== null) {
                graphStatLimitPill.style.display = 'inline-flex';
                graphStatLimit.textContent = `${sensorInfo.limit_min} - ${sensorInfo.limit_max} ${sensorInfo.unit}`;
            } else if (sensorInfo.limit_max !== null) {
                graphStatLimitPill.style.display = 'inline-flex';
                graphStatLimit.textContent = `< ${sensorInfo.limit_max} ${sensorInfo.unit}`;
            } else if (sensorInfo.limit_min !== null) {
                graphStatLimitPill.style.display = 'inline-flex';
                graphStatLimit.textContent = `> ${sensorInfo.limit_min} ${sensorInfo.unit}`;
            } else {
                graphStatLimitPill.style.display = 'none';
            }
        }

        // Prepare labels and datasets
        const labels = points.map(p => p.t);
        const baselineData = points.map(p => p.y);
        const anomalyData = points.map(p => (p.status === 'anomaly' ? p.y : null));
        const suspectData = points.map(p => (p.status === 'suspect' ? p.y : null));

        const datasets = [
            {
                label: `Operating Baseline (${sensorInfo.unit})`,
                data: baselineData,
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.08)',
                borderWidth: 1.8,
                fill: true,
                tension: 0.15,
                pointRadius: points.length < 80 ? 3 : 0,
                pointHoverRadius: 6,
                pointBackgroundColor: '#06b6d4',
                order: 3,
            },
            {
                label: `Anomaly Outliers (Faults)`,
                data: anomalyData,
                borderColor: '#ef4444',
                backgroundColor: '#ef4444',
                showLine: false,
                pointRadius: 5.5,
                pointHoverRadius: 8,
                pointBorderColor: '#ffffff',
                pointBorderWidth: 1.5,
                pointStyle: 'circle',
                order: 1,
            },
            {
                label: `Sensor Suspect (Spike / Defect)`,
                data: suspectData,
                borderColor: '#94a3b8',
                backgroundColor: '#64748b',
                showLine: false,
                pointRadius: 6.5,
                pointHoverRadius: 9,
                pointBorderColor: '#f8fafc',
                pointBorderWidth: 1.5,
                pointStyle: 'rectRot',
                order: 0,
            }
        ];

        // Safe Limit Threshold Overlays
        function createDashedLineIcon(color) {
            const c = document.createElement('canvas');
            c.width = 22;
            c.height = 10;
            const ctx = c.getContext('2d');
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.2;
            ctx.setLineDash([5, 3]);
            ctx.beginPath();
            ctx.moveTo(0, 5);
            ctx.lineTo(22, 5);
            ctx.stroke();
            return c;
        }

        if (sensorInfo.limit_max !== null && points.length > 0) {
            datasets.push({
                label: `Upper Safe Limit (${sensorInfo.limit_max} ${sensorInfo.unit})`,
                data: new Array(points.length).fill(sensorInfo.limit_max),
                borderColor: 'rgba(239, 68, 68, 0.75)',
                borderDash: [6, 4],
                borderWidth: 1.5,
                pointRadius: 0,
                pointHoverRadius: 0,
                pointStyle: createDashedLineIcon('#ef4444'),
                fill: false,
                order: 4,
            });
        }
        if (sensorInfo.limit_min !== null && points.length > 0) {
            datasets.push({
                label: `Lower Safe Limit (${sensorInfo.limit_min} ${sensorInfo.unit})`,
                data: new Array(points.length).fill(sensorInfo.limit_min),
                borderColor: 'rgba(245, 158, 11, 0.75)',
                borderDash: [6, 4],
                borderWidth: 1.5,
                pointRadius: 0,
                pointHoverRadius: 0,
                pointStyle: createDashedLineIcon('#f59e0b'),
                fill: false,
                order: 4,
            });
        }

        if (telemetryChartInstance) {
            telemetryChartInstance.destroy();
            telemetryChartInstance = null;
        }

        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded');
            return;
        }

        const ctx = telemetryCanvas.getContext('2d');
        telemetryChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: points.length > 1000 ? 0 : 350,
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#cbd5e1',
                            font: { family: 'Inter', size: 12, weight: '500' },
                            usePointStyle: true,
                            padding: 16,
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.94)',
                        titleColor: '#f8fafc',
                        bodyColor: '#e2e8f0',
                        borderColor: 'rgba(148, 163, 184, 0.25)',
                        borderWidth: 1,
                        padding: 12,
                        boxPadding: 6,
                        usePointStyle: true,
                        callbacks: {
                            label: function(context) {
                                const idx = context.dataIndex;
                                const pt = points[idx];
                                if (!pt) return `${context.dataset.label}: ${context.parsed.y}`;
                                if (context.datasetIndex === 2 || pt.status === 'suspect') {
                                    return `⚠️ SENSOR SUSPECT: ${pt.y} ${sensorInfo.unit} [${pt.fault || 'Spike'}]`;
                                }
                                if (context.datasetIndex === 1 || pt.status === 'anomaly') {
                                    return `🚨 ANOMALY OUTLIER: ${pt.y} ${sensorInfo.unit} [${pt.fault || 'Fault'}]`;
                                }
                                if (context.datasetIndex === 0) {
                                    const statusText = pt.status === 'suspect' ? 'Sensor Suspect' : (pt.status === 'anomaly' ? 'Anomaly Outlier' : 'Healthy Baseline');
                                    return `Telemetry: ${pt.y} ${sensorInfo.unit} (${statusText})`;
                                }
                                return `${context.dataset.label}: ${context.parsed.y}`;
                            },
                            afterLabel: function(context) {
                                const idx = context.dataIndex;
                                const pt = points[idx];
                                if (pt && pt.asset) {
                                    return `Asset: ${pt.asset}`;
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Timestamp',
                            color: '#94a3b8',
                            font: { family: 'Inter', size: 12, weight: '600' }
                        },
                        ticks: {
                            color: '#64748b',
                            font: { family: 'Inter', size: 11 },
                            maxTicksLimit: 10,
                            maxRotation: 0,
                            autoSkip: true,
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.04)',
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: `${sensorInfo.name} (${sensorInfo.unit})`,
                            color: '#94a3b8',
                            font: { family: 'Inter', size: 12, weight: '600' }
                        },
                        ticks: {
                            color: '#64748b',
                            font: { family: 'Inter', size: 11 },
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                        }
                    }
                }
            }
        });
    }
});
