// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import config from 'config';
import { ImageFilter, ImageFilterAlias } from 'utils/image-processing';

// Pixel filters run in a worker, and the frames next to the one on screen are filtered ahead, so stepping through a
// job does not wait ~1 s per frame (24 MP frames). Results are cached by frame and filter settings.
interface FrameImage {
    renderWidth: number;
    renderHeight: number;
    imageData: ImageBitmap;
}

const CACHE_SIZE = 4; // ~100 MB each: the frame on screen, its neighbours, one spare
const cache = new Map<string, Promise<ImageBitmap>>();
interface PendingRequest {
    resolve(bitmap: ImageBitmap): void;
    reject(error: Error): void;
}
const pending = new Map<number, PendingRequest>();
let worker: Worker | null = null;
let nextId = 0;

function filterInWorker(image: FrameImage, filters: object[]): Promise<ImageBitmap> {
    if (!worker) {
        worker = new Worker(new URL('./pixel-filters.worker', import.meta.url));
        worker.onmessage = ({ data }: MessageEvent) => {
            const request = pending.get(data.id);
            pending.delete(data.id);
            if (data.error) request?.reject(new Error(data.error));
            else request?.resolve(data.bitmap);
        };
    }
    const id = nextId++;
    return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        (worker as Worker).postMessage({
            id,
            bitmap: image.imageData, // copied, not transferred: cvat-core keeps its own
            width: image.renderWidth,
            height: image.renderHeight,
            filters,
            openCVUrl: new URL(config.OPENCV_PATH, window.location.href).href,
        });
    });
}

export function filteredFrame(frame: number, image: FrameImage, filters: ImageFilter[]): Promise<ImageBitmap> {
    const specs = filters.map(({ alias, modifier }) => ({
        alias,
        params: alias === ImageFilterAlias.ENHANCEMENT ? modifier.toJSON().params : null,
    }));
    const key = `${frame}|${JSON.stringify(specs)}`;
    const result = cache.get(key) ?? filterInWorker(image, specs);
    cache.delete(key); // re-inserted as the most recent
    cache.set(key, result);
    result.catch(() => cache.delete(key));
    if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value as string);
    return result;
}

// the frames before and after, filtered in the background; errors surface when the frame is actually shown
export function prefetchNeighbours(job: any, frame: number, filters: ImageFilter[]): void {
    [frame + 1, frame - 1].filter((n) => n >= job.startFrame && n <= job.stopFrame).forEach((n) => {
        job.frames.get(n)
            .then((frameData: any) => frameData.data())
            .then((image: FrameImage) => filteredFrame(n, image, filters))
            .catch(() => {});
    });
}
