'use client';

import { useRef, useState, useLayoutEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { useSceneSelector } from '@/lib/contexts/scene-context';
import { useCanvasStore } from '@/lib/store/canvas';
import type { SlideContent } from '@/lib/types/stage';
import type { PPTElement } from '@/lib/types/slides';

interface SpotlightRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Spotlight overlay component
 *
 * Uses DOM measurement (getBoundingClientRect) to compute spotlight position,
 * avoiding alignment offsets from percentage coordinate conversion.
 */
export function SpotlightOverlay() {
  const spotlightElementId = useCanvasStore.use.spotlightElementId();
  const spotlightOptions = useCanvasStore.use.spotlightOptions();
  const containerRef = useRef<HTMLDivElement>(null);
  const [rect, setRect] = useState<SpotlightRect | null>(null);

  const elements = useSceneSelector<SlideContent, PPTElement[]>(
    (content) => content.canvas.elements,
  );

  // Compute target element position in SVG coordinate system via DOM measurement
  const measure = useCallback(() => {
    if (!spotlightElementId || !containerRef.current) {
      setRect(null);
      return;
    }

    const domElement = document.getElementById(`screen-element-${spotlightElementId}`);
    if (!domElement) {
      setRect(null);
      return;
    }

    // Prefer measuring .element-content (the actual rendered area for auto-height)
    const contentEl = domElement.querySelector('.element-content');
    const targetEl = contentEl ?? domElement;

    const containerRect = containerRef.current.getBoundingClientRect();
    const targetRect = targetEl.getBoundingClientRect();

    if (containerRect.width === 0 || containerRect.height === 0) {
      setRect(null);
      return;
    }

    // Convert to SVG viewBox 0-100 coordinates
    setRect({
      x: ((targetRect.left - containerRect.left) / containerRect.width) * 100,
      y: ((targetRect.top - containerRect.top) / containerRect.height) * 100,
      w: (targetRect.width / containerRect.width) * 100,
      h: (targetRect.height / containerRect.height) * 100,
    });
  }, [spotlightElementId]);

  useLayoutEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- DOM measurement requires effect
    measure();
  }, [measure, elements]);

  const active = !!spotlightElementId && !!spotlightOptions && !!rect;
  const dimness = spotlightOptions?.dimness ?? 0.7;

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-[100] pointer-events-none overflow-hidden"
    >
      <AnimatePresence mode="wait">
        {active && rect && (
          <motion.div
            key={`spotlight-${spotlightElementId}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0"
          >
            {/* Top dim zone */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute left-0 right-0 top-0"
              style={{
                height: `${Math.max(rect.y - 0.6, 0)}%`,
                backgroundColor: `rgba(0,0,0,${dimness})`,
              }}
            />
            {/* Bottom dim zone */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute left-0 right-0"
              style={{
                top: `${Math.min(rect.y + rect.h + 0.6, 100)}%`,
                bottom: 0,
                backgroundColor: `rgba(0,0,0,${dimness})`,
              }}
            />
            {/* Left dim zone */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute"
              style={{
                top: `${Math.max(rect.y - 0.6, 0)}%`,
                height: `${Math.min(rect.h + 1.2, 100)}%`,
                left: 0,
                width: `${Math.max(rect.x - 0.4, 0)}%`,
                backgroundColor: `rgba(0,0,0,${dimness})`,
              }}
            />
            {/* Right dim zone */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute"
              style={{
                top: `${Math.max(rect.y - 0.6, 0)}%`,
                height: `${Math.min(rect.h + 1.2, 100)}%`,
                right: 0,
                width: `${Math.max(100 - (rect.x + rect.w + 0.4), 0)}%`,
                backgroundColor: `rgba(0,0,0,${dimness})`,
              }}
            />

            {/* Focus border */}
            <motion.div
              initial={{
                left: `${Math.max(rect.x - 4, 0)}%`,
                top: `${Math.max(rect.y - 4, 0)}%`,
                width: `${Math.min(rect.w + 8, 100)}%`,
                height: `${Math.min(rect.h + 8, 100)}%`,
                opacity: 0,
              }}
              animate={{
                left: `${Math.max(rect.x - 0.4, 0)}%`,
                top: `${Math.max(rect.y - 0.6, 0)}%`,
                width: `${Math.min(rect.w + 0.8, 100)}%`,
                height: `${Math.min(rect.h + 1.2, 100)}%`,
                opacity: 1,
              }}
              transition={{
                duration: 0.5,
                delay: 0.05,
                ease: [0.16, 1, 0.3, 1],
              }}
              className="absolute rounded-[6px] border border-white/70"
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
