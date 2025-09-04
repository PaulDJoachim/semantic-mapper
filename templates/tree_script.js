const treeData = {tree_data};

class ClusterVisualizer {{
    constructor(canvas, options = {{}}) {{
        this.canvas = canvas;
        this.options = {{
            autoRotate: options.autoRotate !== false,
            fullscreen: options.fullscreen === true,
            ...options
        }};

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.rayGroup = null;
        this.sphereGroup = null;
        this.isRotating = this.options.autoRotate;
        this.currentData = null;

        this.init();
    }}

    init() {{
        if (!window.THREE) {{
            console.warn('Three.js not available');
            return;
        }}

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);

        const rect = this.canvas.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(60, rect.width / rect.height, 0.1, 1000);

        this.resetCamera();

        this.renderer = new THREE.WebGLRenderer({{ canvas: this.canvas, antialias: true }});
        this.renderer.setSize(rect.width, rect.height);

        const ambient = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 0.8);
        directional.position.set(1, 1, 1);
        this.scene.add(directional);

        const axes = new THREE.AxesHelper(0.8);
        this.scene.add(axes);

        const originGeo = new THREE.SphereGeometry(0.02, 16, 16);
        const originMat = new THREE.MeshBasicMaterial({{ color: 0xffffff }});
        const origin = new THREE.Mesh(originGeo, originMat);
        this.scene.add(origin);

        this.rayGroup = new THREE.Group();
        this.sphereGroup = new THREE.Group();
        this.scene.add(this.rayGroup);
        this.scene.add(this.sphereGroup);

        this.setupControls();
        this.animate();
    }}

    resetCamera() {{
        this.camera.position.set(.8, 0.6, .8);
        this.camera.lookAt(0, 0, 0);
    }}

    setupControls() {{
        if (!this.renderer) return;

        let isDragging = false;
        let previousMouse = {{ x: 0, y: 0 }};

        this.canvas.addEventListener('mousedown', (e) => {{
            isDragging = false;
            this.isRotating = false;
            previousMouse = {{ x: e.offsetX, y: e.offsetY }};
        }});

        this.canvas.addEventListener('mouseup', (e) => {{
            this.isRotating = true;
            previousMouse = {{ x: e.offsetX, y: e.offsetY }};
        }});

        this.canvas.addEventListener('mousemove', (e) => {{
            if (e.buttons === 1) {{
                isDragging = true;
                const deltaMove = {{
                    x: e.offsetX - previousMouse.x,
                    y: e.offsetY - previousMouse.y
                }};

                const spherical = new THREE.Spherical();
                spherical.setFromVector3(this.camera.position);

                spherical.theta -= deltaMove.x * 0.01;
                spherical.phi += deltaMove.y * 0.01;

                spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                this.camera.position.setFromSpherical(spherical);
                this.camera.lookAt(0, 0, 0);
            }}
            previousMouse = {{ x: e.offsetX, y: e.offsetY }};
        }});

        this.canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const distance = this.camera.position.length();
            const newDistance = Math.max(0.5, Math.min(5, distance + e.deltaY * 0.002));
            this.camera.position.normalize().multiplyScalar(newDistance);
        }});
    }}

    showClusters(clusterData) {{
        this.currentData = clusterData;
        this.clear();

        if (!clusterData || !clusterData.samples) return;

        clusterData.samples.forEach(sample => {{
            const direction = new THREE.Vector3(...sample.direction);
            const endPoint = direction.multiplyScalar(1);

            const rayGeo = new THREE.BufferGeometry();
            rayGeo.setFromPoints([new THREE.Vector3(0, 0, 0), endPoint]);

            const rayMat = new THREE.LineBasicMaterial({{
                color: sample.color,
                opacity: 0.8,
                transparent: true
            }});

            const ray = new THREE.Line(rayGeo, rayMat);
            this.rayGroup.add(ray);

            const sphereGeo = new THREE.SphereGeometry(0.01, 8, 8);
            const sphereMat = new THREE.MeshPhongMaterial({{ color: sample.color }});
            const sphere = new THREE.Mesh(sphereGeo, sphereMat);
            sphere.position.copy(endPoint);
            this.sphereGroup.add(sphere);
        }});

        this.resetCamera();
    }}

    clear() {{
        this.rayGroup.clear();
        this.sphereGroup.clear();
    }}

    animate() {{
        if (!this.renderer) return;

        requestAnimationFrame(() => this.animate());

        if (this.isRotating && this.rayGroup.children.length > 0) {{
            this.rayGroup.rotation.y += 0.005;
            this.sphereGroup.rotation.y += 0.005;
        }}

        this.renderer.render(this.scene, this.camera);
    }}
}}

class TreeVisualizer {{
    constructor(canvas, treeData) {{
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.treeData = treeData;
        this.selectedNode = null;
        this.clusterViewer = null;

        this.viewState = {{
            offsetX: 0,
            offsetY: 0,
            zoom: 1,
            nodeScale: 1,
            layout: 'radial'
        }};

        this.nodes = [];
        this.edges = [];
        this.expandedNodes = new Set();
        this.nodeDimensionsCache = new Map();

        // Path highlighting state
        this.highlightedNodes = new Set();
        this.highlightedEdges = new Set();

        this.setupCanvas();
        this.setupEventListeners();
        this.processTree();
        this.render();
    }}

    setupCanvas() {{
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }}

    resizeCanvas() {{
        const container = document.getElementById('tree-container');
        const controls = document.querySelector('.controls');
        const availableWidth = container.clientWidth;
        const availableHeight = container.clientHeight - controls.offsetHeight - 40;

        this.canvas.width = availableWidth;
        this.canvas.height = availableHeight;

        this.viewState.offsetX = availableWidth / 2;
        this.viewState.offsetY = availableHeight / 2;

        this.nodeDimensionsCache.clear();
        this.render();
    }}

    initClusterViewer() {{
        if (!this.clusterViewer) {{
            const canvas3d = document.getElementById('cluster-viewer');
            this.clusterViewer = new ClusterVisualizer(canvas3d, {{
                fullscreen: false,
                autoRotate: true
            }});
        }}
        return this.clusterViewer;
    }}

    setupEventListeners() {{
        let isDragging = false;
        let lastX, lastY;
        let dragStarted = false;

        this.canvas.addEventListener('mousedown', (e) => {{
            isDragging = true;
            dragStarted = false;
            lastX = e.clientX;
            lastY = e.clientY;
        }});

        this.canvas.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const deltaX = e.clientX - lastX;
                const deltaY = e.clientY - lastY;

                if (Math.abs(deltaX) > 3 || Math.abs(deltaY) > 3) {{
                    dragStarted = true;
                }}

                this.viewState.offsetX += deltaX;
                this.viewState.offsetY += deltaY;
                lastX = e.clientX;
                lastY = e.clientY;
                this.render();
            }}
        }});

        this.canvas.addEventListener('mouseup', (e) => {{
            if (isDragging && !dragStarted) {{
                this.handleClick(e);
            }}
            isDragging = false;
            dragStarted = false;
        }});

        this.canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.1, Math.min(8, this.viewState.zoom * zoomFactor));

            if (newZoom !== this.viewState.zoom) {{
                this.viewState.zoom = newZoom;
                document.getElementById('zoomSlider').value = newZoom;
                document.getElementById('zoomValue').textContent = newZoom.toFixed(1);
                this.nodeDimensionsCache.clear();
                this.render();
            }}
        }});

        document.getElementById('zoomSlider').addEventListener('input', (e) => {{
            this.viewState.zoom = parseFloat(e.target.value);
            document.getElementById('zoomValue').textContent = this.viewState.zoom.toFixed(1);
            this.nodeDimensionsCache.clear();
            this.render();
        }});

        document.getElementById('nodeSizeSlider').addEventListener('input', (e) => {{
            this.viewState.nodeScale = parseFloat(e.target.value);
            document.getElementById('nodeSizeValue').textContent = this.viewState.nodeScale.toFixed(1);
            this.nodeDimensionsCache.clear();
            this.processTree();
            this.render();
        }});

        document.getElementById('layoutSelect').addEventListener('change', (e) => {{
            this.viewState.layout = e.target.value;
            this.processTree();
            this.render();
        }});

        document.getElementById('resetView').addEventListener('click', () => {{
            this.viewState.offsetX = this.canvas.width / 2;
            this.viewState.offsetY = this.canvas.height / 2;
            this.viewState.zoom = 1;
            document.getElementById('zoomSlider').value = 1;
            document.getElementById('zoomValue').textContent = '1.0';
            this.nodeDimensionsCache.clear();
            this.render();
        }});
    }}

    handleClick(e) {{
        const rect = this.canvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;

        let nodeClicked = false;

        for (const node of this.nodes) {{
            const screen = this.worldToScreen(node.x, node.y);
            const dimensions = this.getNodeDimensions(node);

            const left = screen.x - dimensions.width / 2;
            const right = screen.x + dimensions.width / 2;
            const top = screen.y - dimensions.height / 2;
            const bottom = screen.y + dimensions.height / 2;

            if (clickX >= left && clickX <= right && clickY >= top && clickY <= bottom) {{
                nodeClicked = true;

                if (dimensions.canExpand) {{
                    if (this.expandedNodes.has(node.id)) {{
                        this.expandedNodes.delete(node.id);
                    }} else {{
                        this.expandedNodes.add(node.id);
                    }}
                    this.nodeDimensionsCache.clear();
                    this.render();
                }}

                this.selectNode(node);
                break;
            }}
        }}

        // If no node was clicked, clear selection
        if (!nodeClicked) {{
            this.clearSelection();
        }}
    }}

    clearSelection() {{
        this.selectedNode = null;
        this.highlightedNodes.clear();
        this.highlightedEdges.clear();
        this.hideClusterData();
        this.render();
    }}

    selectNode(node) {{
        this.selectedNode = node;
        this.updateHighlightedPaths(node);
        this.render();

        // Always show the selection data (path info), regardless of cluster data
        this.showSelectionData(node);
    }}

    // get the complete path from root to a given node
    getPathToNode(nodeId) {{
        const path = [];
        let currentNode = this.nodes[nodeId];

        // Traverse up the tree to collect all tokens
        while (currentNode) {{
            path.unshift(currentNode.text); // Add to beginning of array

            if (currentNode.parentId !== null) {{
                currentNode = this.nodes[currentNode.parentId];
            }} else {{
                break;
            }}
        }}

        return path;
    }}

    // get path with selected node info
    getPathWithSelection(nodeId) {{
        const pathTokens = this.getPathToNode(nodeId);
        const selectedToken = this.nodes[nodeId].text;

        return {{
            fullPath: pathTokens.join(''),
            selectedToken: selectedToken,
            pathBeforeSelected: pathTokens.slice(0, -1).join('')
        }};
    }}

    // Path highlighting methods
    updateHighlightedPaths(node) {{
        this.highlightedNodes.clear();
        this.highlightedEdges.clear();

        if (!node) return;

        // Get all connected nodes (ancestors + descendants)
        const connectedNodes = new Set();

        // Add ancestors (path to root)
        this.getPathToRoot(node.id, connectedNodes);

        // Add descendants (all children)
        this.getAllDescendants(node.id, connectedNodes);

        this.highlightedNodes = connectedNodes;

        // Find edges connecting highlighted nodes
        this.edges.forEach((edge, index) => {{
            if (this.highlightedNodes.has(edge.from) && this.highlightedNodes.has(edge.to)) {{
                this.highlightedEdges.add(index);
            }}
        }});
    }}

    getPathToRoot(nodeId, connectedNodes) {{
        connectedNodes.add(nodeId);
        const node = this.nodes[nodeId];
        if (node && node.parentId !== null) {{
            this.getPathToRoot(node.parentId, connectedNodes);
        }}
    }}

    getAllDescendants(nodeId, connectedNodes) {{
        connectedNodes.add(nodeId);

        // Find all edges where this node is the parent
        this.edges.forEach(edge => {{
            if (edge.from === nodeId && !connectedNodes.has(edge.to)) {{
                this.getAllDescendants(edge.to, connectedNodes);
            }}
        }});
    }}

    showSelectionData(node) {{
        document.getElementById('no-selection').classList.add('hidden');
        document.getElementById('cluster-content').classList.remove('hidden');

        // Get path information with selection details
        const pathInfo = this.getPathWithSelection(node.id);

        // Create HTML with bold selected token
        const selectedNodeElement = document.getElementById('selected-node-text');
        selectedNodeElement.innerHTML = `"${{pathInfo.pathBeforeSelected}}<strong>${{pathInfo.selectedToken}}</strong>"`;

        // Check if node has cluster data
        const hasClusterData = node.cluster_data && node.cluster_data.samples && node.cluster_data.samples.length > 0;

        if (hasClusterData) {{
            // Show cluster statistics
            const stats = node.cluster_data.stats || {{}};
            const statsHtml = Object.entries(stats)
                .map(([cluster, count]) => `<div>${{cluster}}: ${{count}} samples</div>`)
                .join('');
            document.getElementById('cluster-stats').innerHTML = `<div class="cluster-info">${{statsHtml}}</div>`;

            // Show cluster samples
            const clusterGroups = {{}};
            node.cluster_data.samples.forEach(sample => {{
                const clusterId = sample.cluster;
                const clusterKey = clusterId === -1 ? 'noise' : `cluster_${{clusterId}}`;
                if (!clusterGroups[clusterKey]) {{
                    clusterGroups[clusterKey] = [];
                }}
                clusterGroups[clusterKey].push(sample);
            }});

            const clustersHtml = Object.entries(clusterGroups)
                .sort(([a], [b]) => {{
                    if (a === 'noise') return 1;
                    if (b === 'noise') return -1;
                    return a.localeCompare(b);
                }})
                .map(([clusterKey, samples]) => {{
                    const displayName = clusterKey === 'noise' ? 'Noise' : clusterKey.replace('_', ' ');
                    const sampleCount = samples.length;

                    const samplesHtml = samples
                        .map(sample => `<div class="sample-text" style="border-left-color: ${{sample.color}}">${{sample.text}}</div>`)
                        .join('');

                    return `
                        <div class="cluster-group">
                            <div class="cluster-header" onclick="toggleCluster('${{clusterKey}}')">
                                <span>${{displayName}} (${{sampleCount}} samples)</span>
                                <span class="expand-icon" id="icon-${{clusterKey}}">▼</span>
                            </div>
                            <div class="cluster-content" id="content-${{clusterKey}}">
                                ${{samplesHtml}}
                            </div>
                        </div>
                    `;
                }})
                .join('');

            document.getElementById('cluster-samples').innerHTML = clustersHtml;

            // Show 3D cluster visualization
            const viewer = this.initClusterViewer();
            viewer.showClusters(node.cluster_data);
        }} else {{
            // Node has no cluster data - show basic info
            document.getElementById('cluster-stats').innerHTML = '<div class="cluster-info">No cluster data available for this node.</div>';
            document.getElementById('cluster-samples').innerHTML = '';

            // Clear the 3D viewer
            if (this.clusterViewer) {{
                this.clusterViewer.clear();
            }}
        }}
    }}

    hideClusterData() {{
        document.getElementById('no-selection').classList.add('hidden');
        document.getElementById('cluster-content').classList.add('hidden');
        if (this.clusterViewer) {{
            this.clusterViewer.clear();
        }}
    }}

    screenToWorld(screenX, screenY) {{
        return {{
            x: (screenX - this.viewState.offsetX) / this.viewState.zoom,
            y: (screenY - this.viewState.offsetY) / this.viewState.zoom
        }};
    }}

    worldToScreen(x, y) {{
        return {{
            x: (x * this.viewState.zoom) + this.viewState.offsetX,
            y: (y * this.viewState.zoom) + this.viewState.offsetY
        }};
    }}

    processTree() {{
        this.nodes = [];
        this.edges = [];

        if (this.viewState.layout === 'radial') {{
            this.layoutRadial();
        }} else {{
            this.layoutTree();
        }}

        // Update highlighting for current selection
        if (this.selectedNode) {{
            this.updateHighlightedPaths(this.selectedNode);
        }}
    }}

    layoutRadial() {{
        this.traverseRadial(this.treeData, 0, 0, 0, null, 0);
    }}

    layoutTree() {{
        this.traverseTree(this.treeData, 0, -200, null, 0);
    }}

    traverseRadial(node, x, y, parentAngle, parentId, level) {{
        const nodeId = this.nodes.length;

        this.nodes.push({{
            id: nodeId,
            x: x,
            y: y,
            text: node.token_text,
            proportion: node.proportion,
            depth: node.depth,
            level: level,
            parentId: parentId,
            cluster_data: node.cluster_data
        }});

        if (parentId !== null) {{
            this.edges.push({{ from: parentId, to: nodeId }});
        }}

        if (node.children && node.children.length > 0) {{
            const childCount = node.children.length;
            const baseRadius = 80 + (level * level * 10); // Quadratic growth
            const effectiveRadius = baseRadius * this.viewState.nodeScale;
            const levelFactor = Math.pow(0.6, level-1); // Tighter at deeper levels
            const maxAngleSpread = level === 0 ? Math.PI * 1.5 : Math.PI * 0.9;
            const minAngleSpread = 0.5;
            const angleSpread = Math.min(maxAngleSpread, Math.max(minAngleSpread, childCount * 0.2 * levelFactor));
            const startAngle = parentAngle - angleSpread / 2;

            node.children.forEach((child, index) => {{
                let childAngle = childCount === 1 ? parentAngle :
                    startAngle + (index * angleSpread / (childCount - 1));

                const childX = x + Math.cos(childAngle) * effectiveRadius;
                const childY = y + Math.sin(childAngle) * effectiveRadius;

                this.traverseRadial(child, childX, childY, childAngle, nodeId, level + 1);
            }});
        }}
    }}

    traverseTree(node, x, y, parentId, level) {{
        const nodeId = this.nodes.length;

        this.nodes.push({{
            id: nodeId,
            x: x,
            y: y,
            text: node.token_text,
            proportion: node.proportion,
            depth: node.depth,
            level: level,
            parentId: parentId,
            cluster_data: node.cluster_data
        }});

        if (parentId !== null) {{
            this.edges.push({{ from: parentId, to: nodeId }});
        }}

        if (node.children && node.children.length > 0) {{
            const childSpacing = 160 * this.viewState.nodeScale;
            const totalWidth = (node.children.length - 1) * childSpacing;
            const startX = x - totalWidth / 2;
            const verticalSpacing = 140 * this.viewState.nodeScale;

            node.children.forEach((child, index) => {{
                const childX = startX + (index * childSpacing);
                const childY = y + verticalSpacing;
                this.traverseTree(child, childX, childY, nodeId, level + 1);
            }});
        }}
    }}

    wrapText(text, maxWidth, fontSize) {{
        this.ctx.font = `${{fontSize}}px Arial`;

        const words = text.trim().split(/\s+/);
        const lines = [];
        let currentLine = '';

        for (const word of words) {{
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            const testWidth = this.ctx.measureText(testLine).width;

            if (testWidth <= maxWidth || !currentLine) {{
                currentLine = testLine;
            }} else {{
                lines.push(currentLine);
                currentLine = word;
            }}
        }}

        if (currentLine) {{
            lines.push(currentLine);
        }}

        return lines;
    }}

    getNodeDimensions(node) {{
        const isExpanded = this.expandedNodes.has(node.id);
        const cacheKey = `${{node.id}}-${{this.viewState.zoom}}-${{this.viewState.nodeScale}}-${{isExpanded}}`;

        if (this.nodeDimensionsCache.has(cacheKey)) {{
            return this.nodeDimensionsCache.get(cacheKey);
        }}

        const baseScale = this.viewState.nodeScale;
        const zoomScale = Math.max(0.3, Math.min(1.2, this.viewState.zoom));
        const fontSize = Math.max(8, 10 * baseScale * zoomScale * this.viewState.nodeScale);
        const padding = Math.max(8, 12 * baseScale);

        this.ctx.font = `${{fontSize}}px Arial`;

        const displayText = node.text.trim();
        const baseWidth = Math.max(120, 140 * baseScale);
        const maxTextWidth = baseWidth - (padding * 2);

        const allLines = this.wrapText(displayText, maxTextWidth, fontSize);

        let lines, canExpand;
        const maxCollapsedLines = 3;

        if (!isExpanded && allLines.length > maxCollapsedLines) {{
            lines = allLines.slice(0, maxCollapsedLines);
            lines[maxCollapsedLines - 1] += '...';
            canExpand = true;
        }} else {{
            lines = allLines;
            canExpand = allLines.length > maxCollapsedLines;
        }}

        const width = baseWidth;
        const lineHeight = fontSize + (fontSize * 0.2);
        const height = Math.max(
            (lines.length * lineHeight) + padding * 2,
            40 * baseScale
        );

        const dimensions = {{
            width,
            height,
            padding,
            fontSize,
            lines,
            isExpanded,
            canExpand,
            lineHeight
        }};

        this.nodeDimensionsCache.set(cacheKey, dimensions);
        return dimensions;
    }}

    // Enhanced render method with layered drawing
    render() {{
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw in layers: dimmed elements first, then highlighted on top
        this.drawEdges(false); // Draw non-highlighted edges
        this.drawNodes(false); // Draw non-highlighted nodes
        this.drawEdges(true);  // Draw highlighted edges
        this.drawNodes(true);  // Draw highlighted nodes
    }}

    drawEdges(highlightedOnly = false) {{
        this.ctx.lineWidth = Math.max(1, 2 * this.viewState.zoom);

        this.edges.forEach((edge, index) => {{
            const isHighlighted = this.highlightedEdges.has(index);

            // Skip based on what we're drawing in this pass
            if (highlightedOnly && !isHighlighted) return;
            if (!highlightedOnly && isHighlighted) return;

            const fromNode = this.nodes[edge.from];
            const toNode = this.nodes[edge.to];

            const fromScreen = this.worldToScreen(fromNode.x, fromNode.y);
            const toScreen = this.worldToScreen(toNode.x, toNode.y);

            // Scale proportion values to a more visible range
            // 0.04 -> ~0.2, 0.3 -> ~0.8, 1.0 -> 1.0
            let alpha = Math.min(1.0, 0.1 + (toNode.proportion * 0.9));
            let lineWidth = Math.max(1, 2 * this.viewState.zoom);

            if (isHighlighted) {{
                alpha = Math.min(1, alpha + 0.2); // Brighter
                lineWidth *= 1.5; // Thicker
            }} else if (this.highlightedNodes.size > 0) {{
                alpha *= 0.3; // Dimmed when something else is highlighted
            }}

            this.ctx.strokeStyle = `rgba(100, 100, 100, ${{alpha}})`;
            this.ctx.lineWidth = lineWidth;

            this.ctx.beginPath();
            if (this.viewState.layout === 'tree') {{
                const midY = fromScreen.y + (toScreen.y - fromScreen.y) * 0.6;
                this.ctx.moveTo(fromScreen.x, fromScreen.y);
                this.ctx.bezierCurveTo(fromScreen.x, midY, toScreen.x, midY, toScreen.x, toScreen.y);
            }} else {{
                this.ctx.moveTo(fromScreen.x, fromScreen.y);
                this.ctx.lineTo(toScreen.x, toScreen.y);
            }}
            this.ctx.stroke();

            // Add proportion label on edge
            if (this.viewState.zoom > 0.5) {{ // Only show labels when zoomed in enough
                const midX = (fromScreen.x + toScreen.x) / 2;
                const midY = (fromScreen.y + toScreen.y) / 2;

                const labelFontSize = Math.max(6, 5 * this.viewState.zoom);
                this.ctx.font = `${{labelFontSize}}px Arial`;
                this.ctx.fillStyle = `rgba(255, 255, 0, ${{Math.min(1, alpha + 0.3)}})`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';

                // Format proportion to 2 decimal places
                const proportionText = toNode.proportion.toFixed(2);
                this.ctx.fillText(proportionText, midX, midY);
            }}
        }});
    }}

    drawNodes(highlightedOnly = false) {{
        this.nodes.forEach(node => {{
            const isHighlighted = this.highlightedNodes.has(node.id);

            // Skip based on what we're drawing in this pass
            if (highlightedOnly && !isHighlighted) return;
            if (!highlightedOnly && isHighlighted) return;

            const screen = this.worldToScreen(node.x, node.y);
            this.drawNode(screen.x, screen.y, node, isHighlighted);
        }});
    }}

    drawNode(x, y, node, isHighlighted = false) {{
        const dimensions = this.getNodeDimensions(node);
        const {{ width, height, padding, fontSize, lines, isExpanded, canExpand, lineHeight }} = dimensions;

        const rectX = x - width / 2;
        const rectY = y - height / 2;
        const cornerRadius = Math.max(6, 10 * this.viewState.nodeScale);

        const hue = (node.level * 45) % 360;
        const saturation = Math.max(40, 70 - node.level * 5);
        let lightness = Math.max(35, 55 - node.level * 3);

        const isSelected = this.selectedNode && this.selectedNode.id === node.id;
        const hasClusterData = node.cluster_data && node.cluster_data.samples && node.cluster_data.samples.length > 0;

        // Adjust appearance based on highlight state
        if (isHighlighted) {{
            lightness = Math.min(75, lightness); // Brighter
        }} else if (this.highlightedNodes.size > 0) {{
            lightness = Math.max(10, lightness - 65); // Dimmer
        }}

        const color = `hsl(${{hue}}, ${{saturation}}%, ${{lightness}}%)`;

        let borderColor = '#cccccc';
        let borderWidth = 1;

        if (isSelected) {{
            borderColor = '#ffffff';
            borderWidth = 3;
        }} else if (hasClusterData) {{
            borderColor = '#00ff88';
            borderWidth = 2;
        }}

        if (isHighlighted) {{
            borderWidth *= 1.5; // Thicker border for highlighted nodes
        }}

        // Draw rounded rectangle background
        this.drawRoundedRect(rectX, rectY, width, height, cornerRadius);
        this.ctx.fillStyle = color;
        this.ctx.fill();

        this.ctx.strokeStyle = borderColor;
        this.ctx.lineWidth = borderWidth;
        this.ctx.stroke();

        if (this.viewState.zoom > 0.2) {{
            // Adjust text opacity based on highlight state
            let textAlpha = 1;
            if (!isHighlighted && this.highlightedNodes.size > 0) {{
                textAlpha = 0.5;
            }}

            this.ctx.fillStyle = `rgba(255, 255, 255, ${{textAlpha}})`;
            this.ctx.font = `${{fontSize}}px Arial`;
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';

            const totalTextHeight = (lines.length - 1) * lineHeight;
            const startY = y - (totalTextHeight / 2);

            lines.forEach((line, index) => {{
                const lineY = startY + (index * lineHeight);
                this.ctx.fillText(line, x, lineY);
            }});

            if (canExpand && !isExpanded && this.viewState.zoom > 0.5) {{
                this.ctx.fillStyle = `rgba(170, 170, 170, ${{textAlpha}})`;
                this.ctx.font = `${{Math.max(8, fontSize * 0.7)}}px Arial`;
                this.ctx.fillText('â‹¯', x + width/2 - 15, y - height/2 + 12);
            }}
        }}
    }}

    drawRoundedRect(x, y, width, height, radius) {{
        this.ctx.beginPath();
        this.ctx.moveTo(x + radius, y);
        this.ctx.lineTo(x + width - radius, y);
        this.ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        this.ctx.lineTo(x + width, y + height - radius);
        this.ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        this.ctx.lineTo(x + radius, y + height);
        this.ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        this.ctx.lineTo(x, y + radius);
        this.ctx.quadraticCurveTo(x, y, x + radius, y);
        this.ctx.closePath();
    }}
}}

const canvas = document.getElementById('canvas');
const visualizer = new TreeVisualizer(canvas, treeData);

// Global function for collapsible clusters
function toggleCluster(clusterKey) {{
    const content = document.getElementById(`content-${{clusterKey}}`);
    const icon = document.getElementById(`icon-${{clusterKey}}`);

    if (content && icon) {{
        content.classList.toggle('collapsed');
        icon.classList.toggle('collapsed');
    }}
}}
