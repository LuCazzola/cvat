// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

// Runs the pixel image filters (RedScrap enhancement, histogram equalization) off the main thread, see pixel-filters.ts

import EnhancementImplementation from 'cvat-core/src/opencv/enhancement';
import HistogramEqualizationImplementation from 'cvat-core/src/opencv/histogram-equalization';

const scope = globalThis as any;
let openCV: Promise<any> | null = null;

function loadOpenCV(url: string): Promise<any> {
    // same handshake as utils/opencv-wrapper on the page: Module.onRuntimeInitialized, then the global cv
    openCV = openCV ?? new Promise((resolve) => {
        scope.Module = { onRuntimeInitialized: () => resolve(scope.cv) };
        scope.importScripts(url);
    });
    return openCV;
}

scope.onmessage = async (event: MessageEvent) => {
    const {
        id, bitmap, width, height, filters, openCVUrl,
    } = event.data;
    try {
        const cv = await loadOpenCV(openCVUrl);
        const ctx = new OffscreenCanvas(width, height).getContext('2d') as OffscreenCanvasRenderingContext2D;
        ctx.drawImage(bitmap, 0, 0, width, height);
        bitmap.close();
        let image = ctx.getImageData(0, 0, width, height);
        for (const { alias, params } of filters) {
            const modifier = alias === 'opencv.enhancement' ?
                new EnhancementImplementation(cv, params) : new HistogramEqualizationImplementation(cv);
            image = modifier.processImage(image, 0);
        }
        const result = await createImageBitmap(image);
        scope.postMessage({ id, bitmap: result }, [result]);
    } catch (error: unknown) {
        scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
    }
};
