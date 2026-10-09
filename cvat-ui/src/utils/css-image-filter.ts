// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import GammaCorrection from 'utils/fabric-wrapper/gamma-correction';
import { ImageFilter, ImageFilterAlias, hasFilter } from 'utils/image-processing';

// Gamma, brightness, contrast and saturation as one CSS filter on the frame: done by the GPU, nothing per frame on the
// main thread. Gamma is an SVG feComponentTransfer (out = in^(1/gamma), in sRGB like the fabric filter it replaces on
// screen), applied before brightness. The other image filters still go through the pixel pipeline (pixelFilters).
const GAMMA_ID = 'redscrap-gamma';

function gammaOf(imageFilters: ImageFilter[]): number {
    const filter = hasFilter(imageFilters, ImageFilterAlias.GAMMA_CORRECTION);
    return filter ? (filter.modifier as GammaCorrection).gamma : 1;
}

export function pixelFilters(imageFilters: ImageFilter[]): ImageFilter[] {
    return imageFilters.filter((filter) => filter.alias !== ImageFilterAlias.GAMMA_CORRECTION);
}

// whether the pixel pipeline has to run again: a pixel filter added, removed or reconfigured (the reducer resets
// currentProcessedImage of the filters it changes)
export function pixelFiltersChanged(previous: ImageFilter[], current: ImageFilter[]): boolean {
    const before = pixelFilters(previous);
    const after = pixelFilters(current);
    return before.length !== after.length ||
        after.some((filter, i) => filter !== before[i] || filter.modifier.currentProcessedImage === null);
}

export function cssImageFilter(
    imageFilters: ImageFilter[], brightness: number, contrast: number, saturation: number,
): string {
    const gamma = gammaOf(imageFilters);
    if (!document.getElementById(GAMMA_ID)) {
        const ns = 'http://www.w3.org/2000/svg';
        const svg = document.createElementNS(ns, 'svg');
        const filter = document.createElementNS(ns, 'filter');
        const transfer = document.createElementNS(ns, 'feComponentTransfer');
        svg.setAttribute('style', 'position: absolute; width: 0; height: 0');
        filter.setAttribute('id', GAMMA_ID);
        filter.setAttribute('color-interpolation-filters', 'sRGB');
        ['R', 'G', 'B'].forEach((c) => {
            const func = document.createElementNS(ns, `feFunc${c}`);
            func.setAttribute('type', 'gamma');
            func.setAttribute('amplitude', '1');
            func.setAttribute('offset', '0');
            transfer.appendChild(func);
        });
        filter.appendChild(transfer);
        svg.appendChild(filter);
        document.body.appendChild(svg);
    }
    document.querySelectorAll(`#${GAMMA_ID} feFuncR, #${GAMMA_ID} feFuncG, #${GAMMA_ID} feFuncB`)
        .forEach((func) => func.setAttribute('exponent', String(1 / gamma)));
    return `${gamma !== 1 ? `url(#${GAMMA_ID}) ` : ''}brightness(${brightness}) contrast(${contrast}) saturate(${saturation})`;
}
