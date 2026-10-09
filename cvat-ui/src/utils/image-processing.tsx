// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { ImageProcessing } from 'cvat-core-wrapper';

export enum ImageFilterAlias {
    HISTOGRAM_EQUALIZATION = 'opencv.histogramEqualization',
    ENHANCEMENT = 'opencv.enhancement',
    GAMMA_CORRECTION = 'fabric.gammaCorrection',
}

export interface ImageFilter {
    modifier: ImageProcessing,
    alias: ImageFilterAlias
}

export function hasFilter(filters: ImageFilter[], alias: ImageFilterAlias): ImageFilter | null {
    const index = filters.findIndex((imageFilter) => imageFilter.alias === alias);
    if (index !== -1) {
        return filters[index];
    }
    return null;
}

interface EnhancementParam { label: string, min: number, max: number, step: number, value: number }
const param = (label: string, min: number, max: number, step: number, value: number): EnhancementParam => ({
    label, min, max, step, value,
});
const clip = [param('clipLimit', 0.5, 10, 0.5, 2)];
const riesz = [
    param('alpha', 0.1, 1, 0.05, 0.4), param('gainHigh', 0.5, 3, 0.1, 1.6), param('gainLow', 0.1, 1.5, 0.1, 0.6),
];

// RedScrap image enhancement methods (cvat-core opencv/enhancement.ts): their sliders, starting from production values
export const ENHANCEMENTS: Record<string, EnhancementParam[]> = {
    clahe_luminance: clip,
    clahe_channels: clip,
    he_luminance: [],
    he_channels: [],
    rek: [param('p', 0.5, 5, 0.1, 2), param('alpha', -1, 5, 0.1, -1)], // alpha < 0: estimated from the image
    riesz,
    riesz_luminance_only: riesz,
    super_color: [param('grid', 5, 50, 1, 20)],
    super_luminance: [param('grid', 5, 50, 1, 20)],
};
