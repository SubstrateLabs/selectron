(args) => {
    // Destructure args with default values
    const {
        viewportExpansion = 0,
    } = args || {}; // Use empty object default if args itself is null/undefined

    // Caching mechanisms
    const DOM_CACHE = {
      boundingRects: new WeakMap(),
      computedStyles: new WeakMap(),
      clearCache: () => {
        DOM_CACHE.boundingRects = new WeakMap();
        DOM_CACHE.computedStyles = new WeakMap();
      }
    };

    // Cache helper functions
    function getCachedBoundingRect(element) {
      if (!element) return null;
      if (DOM_CACHE.boundingRects.has(element)) {
        return DOM_CACHE.boundingRects.get(element);
      }
      const rect = element.getBoundingClientRect();
      if (rect) {
        DOM_CACHE.boundingRects.set(element, rect);
      }
      return rect;
    }

    function getCachedComputedStyle(element) {
      if (!element) return null;
      if (DOM_CACHE.computedStyles.has(element)) {
        return DOM_CACHE.computedStyles.get(element);
      }
      const style = window.getComputedStyle(element);
      if (style) {
        DOM_CACHE.computedStyles.set(element, style);
      }
      return style;
    }

    /**
     * Hash map of DOM nodes indexed by their generated ID.
     */
    const DOM_HASH_MAP = {};
    const ID = { current: 0 };

    function getElementPosition(currentElement) {
      if (!currentElement.parentElement) {
        return 0; // No parent means no siblings
      }
    
      const tagName = currentElement.nodeName.toLowerCase();
      const siblings = Array.from(currentElement.parentElement.children)
        .filter((sib) => sib.nodeName.toLowerCase() === tagName);
    
      if (siblings.length === 1) {
        return 0; // Only element of its type
      }
    
      const index = siblings.indexOf(currentElement) + 1; // 1-based index
      return index;
    }

    /**
     * Returns an XPath tree string for an element relative to the nearest boundary (shadow root, iframe, or document root).
     */
    function getXPathTree(element, stopAtBoundary = true) {
      const segments = [];
      let currentElement = element;
  
      while (currentElement && currentElement.nodeType === Node.ELEMENT_NODE) {
        // Stop if we hit a shadow root host or iframe
        const parentNode = currentElement.parentNode;
        if (
          stopAtBoundary &&
          (parentNode instanceof ShadowRoot || // Stop if parent is ShadowRoot
          (currentElement.tagName && currentElement.tagName.toLowerCase() === 'iframe')) // Stop *at* the iframe itself
        ) {
          // Include the final segment for shadow host or iframe itself
          const position = getElementPosition(currentElement);
          const tagName = currentElement.nodeName.toLowerCase();
          const xpathIndex = position > 0 ? `[${position}]` : "";
          segments.unshift(`${tagName}${xpathIndex}`);
          break; 
        }
        // Stop if parent is null or not an element (reached document or similar)
        if (!parentNode || parentNode.nodeType !== Node.ELEMENT_NODE) {
            // Handle case where element is direct child of document or shadow root
            const position = getElementPosition(currentElement);
            const tagName = currentElement.nodeName.toLowerCase();
            const xpathIndex = position > 0 ? `[${position}]` : "";
            segments.unshift(`${tagName}${xpathIndex}`);
            break;
        }

        const position = getElementPosition(currentElement);
        const tagName = currentElement.nodeName.toLowerCase();
        const xpathIndex = position > 0 ? `[${position}]` : "";
        segments.unshift(`${tagName}${xpathIndex}`);
  
        currentElement = parentNode; // Move up the tree
      }
  
      return segments.join("/");
    }

    /**
     * Checks if a text node is considered visible (has geometry and parent is visible).
     */
    function isTextNodeVisible(textNode) {
      try {
        const range = document.createRange();
        range.selectNodeContents(textNode);
        const rects = range.getClientRects();

        if (!rects || rects.length === 0) return false;

        let isAnyRectVisible = false;
        for (const rect of rects) {
          if (rect.width > 0 && rect.height > 0) {
            isAnyRectVisible = true;
            break;
          }
        }
        if (!isAnyRectVisible) return false;

        // Check parent visibility using isElementVisible
        const parentElement = textNode.parentElement;
        return parentElement ? isElementVisible(parentElement) : false;
      } catch (e) {
        console.warn('Error checking text node visibility:', e);
        return false;
      }
    }

    /**
     * Checks if an element is visible (has dimensions and not hidden by CSS).
     */
    function isElementVisible(element) {
      if (!element || element.nodeType !== Node.ELEMENT_NODE) return false;
      const rect = getCachedBoundingRect(element); // Get bounding rect
      const style = getCachedComputedStyle(element);
      return (
        (
          (element.offsetWidth > 0 && element.offsetHeight > 0) || // Original check
          (rect && rect.width > 0 && rect.height > 0) // Added check based on bounding rect
        ) &&
        style && // Ensure style is not null
        style.visibility !== "hidden" &&
        style.display !== "none"
      );
    }

    /**
     * Checks if an element is interactive based on tag, role, attributes, and computed style.
     */
    function isInteractiveElement(element) {
      if (!element || element.nodeType !== Node.ELEMENT_NODE) return false;

      const interactiveCursors = new Set(['pointer', 'move', 'text', 'grab', 'grabbing', 'cell', 'copy', 'alias', 'all-scroll', 'col-resize', 'context-menu', 'crosshair', 'e-resize', 'ew-resize', 'help', 'n-resize', 'ne-resize', 'nesw-resize', 'ns-resize', 'nw-resize', 'nwse-resize', 'row-resize', 's-resize', 'se-resize', 'sw-resize', 'vertical-text', 'w-resize', 'zoom-in', 'zoom-out']);
      const nonInteractiveCursors = new Set(['not-allowed', 'no-drop', 'wait', 'progress', 'initial', 'inherit']);

      function doesElementHaveInteractivePointer(el) {
        if (el.tagName.toLowerCase() === "html") return false;
        const style = getCachedComputedStyle(el);
        return style && interactiveCursors.has(style.cursor);
      }

      if (doesElementHaveInteractivePointer(element)) return true;

      const interactiveTags = new Set(["a", "button", "input", "select", "textarea", "details", "summary", "label", "option", "optgroup", "fieldset", "legend"]);
      const disableAttributes = new Set(['disabled', 'readonly']);

      const tagName = element.tagName.toLowerCase();
      if (interactiveTags.has(tagName)) {
        const style = getCachedComputedStyle(element);
        if (style && nonInteractiveCursors.has(style.cursor)) return false;

        for (const disableAttr of disableAttributes) {
          if (element.hasAttribute(disableAttr) || element[disableAttr]) return false;
        }
        if (element.inert) return false;

        return true;
      }

      // Role checks
      const role = element.getAttribute("role");
      const interactiveRoles = new Set(['button', 'menuitem', 'menuitemradio', 'menuitemcheckbox', 'radio', 'checkbox', 'tab', 'switch', 'slider', 'spinbutton', 'combobox', 'searchbox', 'textbox', 'option', 'scrollbar']);
      if (role && interactiveRoles.has(role)) return true;

      // Class/attribute heuristics
      if (element.classList && (element.classList.contains("button") || element.classList.contains('dropdown-toggle')) ||
          element.getAttribute('data-index') || element.getAttribute('data-toggle') === 'dropdown' ||
          element.getAttribute('aria-haspopup') === 'true') {
        return true;
      }

      // Event listener checks (basic attribute fallback)
      try {
        if (typeof getEventListeners === 'function') {
          const listeners = getEventListeners(element);
          const mouseEvents = ['click', 'mousedown', 'mouseup', 'dblclick'];
          for (const eventType of mouseEvents) {
            if (listeners[eventType] && listeners[eventType].length > 0) return true;
          }
        } else {
          const commonMouseAttrs = ['onclick', 'onmousedown', 'onmouseup', 'ondblclick'];
          if (commonMouseAttrs.some(attr => element.hasAttribute(attr))) return true;
        }
      } catch (e) { /* Ignore errors checking listeners */ }

      return false;
    }

    /**
     * Checks if an element is the topmost element at its center point within its document/shadow root context.
     */
    function isTopElement(element) {
      const rects = element.getClientRects();
      if (!rects || rects.length === 0) return false;

      // Use the center of the middle rect for the check
      const midRect = rects[Math.floor(rects.length / 2)];
      const centerX = midRect.left + midRect.width / 2;
      const centerY = midRect.top + midRect.height / 2;

      // Determine the context (document or shadow root)
      let topEl = null;
      const rootNode = element.getRootNode();
      try {
        if (rootNode instanceof ShadowRoot) {
          topEl = rootNode.elementFromPoint(centerX, centerY);
        } else if (element.ownerDocument) {
          topEl = element.ownerDocument.elementFromPoint(centerX, centerY);
        } else {
            return false; // Cannot determine context
        }
      } catch(e) {
          // elementFromPoint can throw if coords are out of viewport, assume true in this edge case?
          // Or perhaps check if the element *is* in the viewport first?
          // For now, let's return true if elementFromPoint fails.
          return true;
      }

      if (!topEl) return false;

      // Check if the element is the top element or an ancestor of it
      let current = topEl;
      while (current) {
        if (current === element) return true;
        // Stop traversal at the boundary of the current root
        if (current.parentNode === rootNode || current === rootNode) break;
        current = current.parentElement;
      }
      return false;
    }

    /**
     * Checks if an element is within the viewport, optionally expanded.
     * Returns true if viewportExpansion is -1 (check disabled).
     */
    function isInExpandedViewport(element, viewportExpansion) {
      if (viewportExpansion === -1) return true; // Check disabled

      const rects = element.getClientRects();
      if (!rects || rects.length === 0) {
        const boundingRect = getCachedBoundingRect(element);
        if (!boundingRect || boundingRect.width === 0 || boundingRect.height === 0) return false;
        return !(
          boundingRect.bottom < -viewportExpansion ||
          boundingRect.top > window.innerHeight + viewportExpansion ||
          boundingRect.right < -viewportExpansion ||
          boundingRect.left > window.innerWidth + viewportExpansion
        );
      }

      // Check if *any* client rect intersects the viewport
      for (const rect of rects) {
        if (rect.width === 0 || rect.height === 0) continue;
        if (!(
          rect.bottom < -viewportExpansion ||
          rect.top > window.innerHeight + viewportExpansion ||
          rect.right < -viewportExpansion ||
          rect.left > window.innerWidth + viewportExpansion
        )) {
          return true;
        }
      }
      return false;
    }

    // --- Content Element Check --- 
    const CONTENT_ELEMENT_TAGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'span', 'label', 'th', 'td', 'dt', 'dd']);
    function isContentElement(element) {
      if (!element || element.nodeType !== Node.ELEMENT_NODE) return false;
      return CONTENT_ELEMENT_TAGS.has(element.tagName.toLowerCase());
    }
    // --- End Content Element Check --- 

    /**
     * Recursively builds a simplified representation of the DOM tree,
     * focusing on visible elements and their properties.
     */
    function buildDomTree(node, parentIframe = null) {

      // Base cases for skipping nodes
      if (!node || (node.nodeType === Node.ELEMENT_NODE && node.id === 'playwright-highlight-container')) {
        return null;
      }
      // Skip non-element/non-text nodes
      if (node.nodeType !== Node.ELEMENT_NODE && node.nodeType !== Node.TEXT_NODE) {
        return null;
      }

      // --- Handle Text Nodes ---
      if (node.nodeType === Node.TEXT_NODE) {
        const textContent = node.textContent.trim();
        if (!textContent) return null;
        // Only include visible text nodes
        if (!isTextNodeVisible(node)) return null;

        const id = `${ID.current++}`;
        DOM_HASH_MAP[id] = {
          type: "TEXT_NODE",
          text: textContent,
          isVisible: true, // Already checked by isTextNodeVisible
        };
        return id;
      }

      // --- Handle Element Nodes --- 
      const nodeData = {
        tagName: node.tagName.toLowerCase(),
        attributes: {},
        xpath: getXPathTree(node, true),
        children: [],
        isContentElement: false,
        isVisible: false,
        isInteractive: false,
        isTopElement: false,
        isInViewport: false,
        shadowRoot: !!node.shadowRoot, // Store if shadow root exists
      };

      // --- Calculate Visibility FIRST --- 
      nodeData.isVisible = isElementVisible(node);

      // --- Determine if Node Should Be Included & Recursed --- 
      let shouldIncludeAndRecurse = nodeData.isVisible;

      // Special case: Always process children of body
      if (nodeData.tagName === 'body') {
        shouldIncludeAndRecurse = true;
      }

      // --- If Included, Calculate Other Properties & Attributes --- 
      if (shouldIncludeAndRecurse) {
          // Calculate other properties ONLY for nodes we are including
          nodeData.isTopElement = isTopElement(node);
          nodeData.isInteractive = isInteractiveElement(node);
          nodeData.isContentElement = isContentElement(node);
          nodeData.isInViewport = isInExpandedViewport(node, viewportExpansion);

          // Fetch attributes
          const attributeNames = node.getAttributeNames?.() || [];
          for (const name of attributeNames) {
            nodeData.attributes[name] = node.getAttribute(name);
          }

          // --- Recurse into Children --- 
          const tagName = node.tagName.toLowerCase(); // Already lowercased

          if (tagName === "iframe") {
            try {
              const iframeDoc = node.contentDocument || node.contentWindow?.document;
              if (iframeDoc && iframeDoc.body) { // Check iframeDoc.body exists
                // Pass iframe element itself as parentIframe context
                const childId = buildDomTree(iframeDoc.body, node); 
                if (childId) nodeData.children.push(childId);
              }
            } catch (e) {
              console.warn("Unable to access or process iframe content:", e);
            }
          } 
          // Handle shadow DOM
          else if (node.shadowRoot) {
            for (const child of node.shadowRoot.childNodes) {
              // Pass null as parentIframe for shadow roots
              const childId = buildDomTree(child, null); 
              if (childId) nodeData.children.push(childId);
            }
          } 
          // Handle regular elements/text nodes
          else {
            for (const child of node.childNodes) {
              // Pass the current parentIframe context down
              const childId = buildDomTree(child, parentIframe); 
              if (childId) nodeData.children.push(childId);
            }
          }

          // --- Add to Map --- 
          const id = `${ID.current++}`;
          DOM_HASH_MAP[id] = nodeData;
          return id;

      } else {
          // Node is not visible, skip it and its children
          return null; 
      }
    }

    // Clear cache before starting
    DOM_CACHE.clearCache();

    // Start building the tree from the document body
    const rootId = buildDomTree(document.body);

    // Return the map and the ID of the root element
    return { rootId, map: DOM_HASH_MAP };
}