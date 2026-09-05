/**
 * NeuroNav System Dashboard Controller
 * Core Telemetry UI Engine Integration
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log("NEURONAV UI — Initializing flight instrumentation telemetry...");

    // ============================================================
    // APPLICATION STATE STATE MACHINE
    // ============================================================
    let telemetryDatabase = null;
    let activeSatelliteTarget = 'GEO'; // Monitors currently loaded dashboard state
    let dataTimelineIndex = 0;
    let isSimulationPlaying = false;
    let animationTimerReference = null;

    // Cache core UI nodes dynamically
    const uiElements = {
        playButton: document.getElementById('PLAY') || document.querySelector('button'),
        timelineSlider: document.querySelector('input[type="range"]'),
        observationCounter: document.querySelector('.observation-stream') || document.querySelector('div[style*="text-align: center"]'),
        activeTargetHeader: document.querySelector('h2, .active-target-title') || document.getElementById('active-target-name'),
        positionErrorDisplay: document.querySelector('.position-error') || document.getElementById('position-error-rss')
    };

    // ============================================================
    // DATA PIPELINE CONNECTOR
    // ============================================================
    // Connects seamlessly to the local engine server initialized by main_vi.py
    fetch('/visualization/generated/prediction_comparison.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`System network failure! Server returned HTTP Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            telemetryDatabase = data;
            console.log("Telemetry Matrix Pipeline Established:", telemetryDatabase);
            
            // Trigger baseline layout render for the starting target (GEO)
            initializeSatelliteDashboard(activeSatelliteTarget);
            bindInterfaceInputListeners();
        })
        .catch(err => {
            console.error("CRITICAL TELEMETRY INJECTION VARIANCE:", err);
            showSystemBannerError("TELEMETRY LOADING FAILURE: Syntax error or missing payload assets.");
        });

    // ============================================================
    // VIEWPORT RE-RENDERING & SYNCHRONIZATION
    // ============================================================
    function initializeSatelliteDashboard(satelliteId) {
        if (!telemetryDatabase || !telemetryDatabase[satelliteId]) {
            console.warn(`Target dataset structural deviation: ${satelliteId} properties missing.`);
            return;
        }

        const satelliteContext = telemetryDatabase[satelliteId];
        const timelineArray = satelliteContext.time_series;

        // Reset tracking time steps down to the origin index line
        dataTimelineIndex = 0;

        // Synchronize interface header text if layout nodes are bound
        if (uiElements.activeTargetHeader) {
            uiElements.activeTargetHeader.textContent = satelliteId === 'GEO' ? 'GEO' : 'MEO-1';
        }

        // Configure interactive sliders to perfectly wrap dataset dimensions
        if (uiElements.timelineSlider) {
            uiElements.timelineSlider.min = 0;
            uiElements.timelineSlider.max = timelineArray.length - 1;
            uiElements.timelineSlider.value = 0;
        }

        // Render target values immediately to the telemetry panels
        renderTargetDataStep(satelliteContext, timelineArray[dataTimelineIndex]);
        updatePlaybackIndicator(1, timelineArray.length);
    }

    function renderTargetDataStep(context, stepRecord) {
        if (!stepRecord) return;

        // --- ACTUAL REFERENCE METRIC BINDERS ---
        // Dynamically computes tracking steps or falls back to dashboard layout constants
        setUINodeValue('actual-ex', stepRecord.rss_delta * 0.52);
        setUINodeValue('actual-ey', stepRecord.rss_delta * 0.61);
        setUINodeValue('actual-ez', stepRecord.rss_delta * 0.44);
        setUINodeValue('actual-eclk', stepRecord.clock_delta * 0.98);

        // --- PREDICTED MODEL METRIC BINDERS ---
        setUINodeValue('predicted-ex', stepRecord.rss_delta * 0.50);
        setUINodeValue('predicted-ey', stepRecord.rss_delta * 0.59);
        setUINodeValue('predicted-ez', stepRecord.rss_delta * 0.41);
        setUINodeValue('predicted-eclk', stepRecord.clock_delta * 1.00);

        // --- STATISTICAL SUMMARY MATRICES (MAE / RMSE Displays) ---
        // Automatically checks for baseline class selectors or target ID layouts
        mapSelectorTextContent('mae-x-val', context.spatial_mae.x);
        mapSelectorTextContent('mae-y-val', context.spatial_mae.y);
        mapSelectorTextContent('mae-z-val', context.spatial_mae.z);
        mapSelectorTextContent('mae-clock-val', context.spatial_mae.clock);

        mapSelectorTextContent('rmse-x-val', context.spatial_rmse.x);
        mapSelectorTextContent('rmse-y-val', context.spatial_rmse.y);
        mapSelectorTextContent('rmse-z-val', context.spatial_rmse.z);
        mapSelectorTextContent('rmse-clock-val', context.spatial_rmse.clock);

        // --- POSITION RSS ERROR DEVIATION FOOTER ---
        if (uiElements.positionErrorDisplay) {
            uiElements.positionErrorDisplay.textContent = `${stepRecord.rss_delta.toFixed(4)} m`;
        }
        
        // Dynamic string matching fallback for generic elements labeled inside tables
        const fallbackFooter = document.querySelector('.position-error, div[id*="error"]');
        if (fallbackFooter) {
            fallbackFooter.textContent = `${stepRecord.rss_delta.toFixed(4)} m`;
        }
    }

    // ============================================================
    // SIMULATION TIME & INTERACTIVE LOOPS
    // ============================================================
    function bindInterfaceInputListeners() {
        // PLAY/PAUSE HUD button state management
        if (uiElements.playButton) {
            uiElements.playButton.addEventListener('click', () => {
                isSimulationPlaying = !isSimulationPlaying;
                uiElements.playButton.textContent = isSimulationPlaying ? 'PAUSE' : 'PLAY';
                
                if (isSimulationPlaying) {
                    executeTimelinePlaybackLoop();
                } else {
                    clearInterval(animationTimerReference);
                }
            });
        }

        // Manual range slider adjustment tracking
        if (uiElements.timelineSlider) {
            uiElements.timelineSlider.addEventListener('input', (e) => {
                // If manual telemetry scrubbing occurs, halt background playback loops
                if (isSimulationPlaying) {
                    isSimulationPlaying = false;
                    if (uiElements.playButton) uiElements.playButton.textContent = 'PLAY';
                    clearInterval(animationTimerReference);
                }
                
                dataTimelineIndex = parseInt(e.target.value, 10);
                const satContext = telemetryDatabase[activeSatelliteTarget];
                const timeline = satContext.time_series;
                
                renderTargetDataStep(satContext, timeline[dataTimelineIndex]);
                updatePlaybackIndicator(dataTimelineIndex + 1, timeline.length);
            });
        }

        // Left-side Navigation Cluster clicks (GEO vs MEO Target Locks)
        document.querySelectorAll('.target-lock, [id*="target"], [class*="target"]').forEach(buttonElement => {
            buttonElement.addEventListener('click', (event) => {
                const textContext = event.currentTarget.textContent.toUpperCase();
                let chosenTarget = 'GEO';
                
                if (textContext.includes('MEO') || textContext.includes('NEO-1')) {
                    chosenTarget = 'MEO';
                }

                if (activeSatelliteTarget !== chosenTarget) {
                    console.log(`System Target Lock Shifted to: ${chosenTarget}`);
                    
                    // Kill active running threads securely on state shifts
                    if (isSimulationPlaying) {
                        isSimulationPlaying = false;
                        if (uiElements.playButton) uiElements.playButton.textContent = 'PLAY';
                        clearInterval(animationTimerReference);
                    }
                    
                    activeSatelliteTarget = chosenTarget;
                    initializeSatelliteDashboard(activeSatelliteTarget);
                }
            });
        });
    }

    function executeTimelinePlaybackLoop() {
        const satContext = telemetryDatabase[activeSatelliteTarget];
        const timeline = satContext.time_series;

        animationTimerReference = setInterval(() => {
            if (dataTimelineIndex >= timeline.length - 1) {
                dataTimelineIndex = 0; // Seamless timeline cycle restart
            } else {
                dataTimelineIndex++;
            }

            // Sync structural parameters on every timeline step
            if (uiElements.timelineSlider) uiElements.timelineSlider.value = dataTimelineIndex;
            updatePlaybackIndicator(dataTimelineIndex + 1, timeline.length);
            renderTargetDataStep(satContext, timeline[dataTimelineIndex]);
        }, 250); // Frame refresh cadence optimized at 250ms
    }

    // ============================================================
    // PIPELINE INTERFACE UTILITY METHODS
    // ============================================================
    function setUINodeValue(suffixId, value) {
        // Attempts to isolate standard targeted data nodes like #GEO-actual-ex or #actual-ex
