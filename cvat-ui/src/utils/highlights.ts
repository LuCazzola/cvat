// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { BaseImageFilter, SerializedImageFilter } from 'cvat-core-wrapper';
import { ImageFilterAlias } from 'utils/image-processing';

export interface HighlightsOptions {
    knee: number; // 0..1, where compression starts
    strength: number; // 0 = off
}

// Identity below the knee, then t / (1 + strength * t) on the part above it: slope 1 at the knee (no visible edge),
// always increasing, white brought down to knee + (1 - knee) / (1 + strength). Lowers glare, keeps its texture.
export function highlightsCurve({ knee, strength }: HighlightsOptions): Uint8ClampedArray {
    const lut = new Uint8ClampedArray(256);
    for (let v = 0; v < 256; v++) {
        const x = v / 255;
        const t = (x - knee) / (1 - knee);
        lut[v] = 255 * (x <= knee ? x : knee + ((1 - knee) * t) / (1 + strength * t));
    }
    return lut;
}

export default class Highlights extends BaseImageFilter {
    public options: HighlightsOptions;

    constructor(options: HighlightsOptions) {
        super();
        this.options = options;
    }

    public configure(options: object): void {
        this.options = options as HighlightsOptions;
    }

    public toJSON(): SerializedImageFilter {
        return { alias: ImageFilterAlias.HIGHLIGHTS, params: this.options };
    }

    public processImage(src: ImageData, frameNumber: number): ImageData {
        const lut = highlightsCurve(this.options);
        const px = new Uint8ClampedArray(src.data);
        for (let i = 0; i < px.length; i += 4) {
            px[i] = lut[px[i]];
            px[i + 1] = lut[px[i + 1]];
            px[i + 2] = lut[px[i + 2]];
        }
        this.currentProcessedImage = frameNumber;
        return new ImageData(px, src.width, src.height);
    }
}
