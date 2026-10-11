// meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.). See LICENSE at the repository root.
// No GPL-licensed source was consulted.
// The domain card names the patches the default view hides, capped at eight with the rest counted.
import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { buildSyntheticSphereInBox } from '../data/syntheticSphereInBox'
import { DomainCard } from './DomainCard'

const manifest = buildSyntheticSphereInBox().manifest
const noop = (): void => {}

function render(locale: 'en' | 'ko', hidden?: string[]): string {
  const props = hidden === undefined ? {} : { hiddenPatches: hidden }
  return renderToStaticMarkup(<DomainCard locale={locale} manifest={manifest} projection="perspective" onToggleProjection={noop} {...props} />)
}

describe('DomainCard hidden patches', () => {
  it('names the hidden patches', () => {
    const html = render('en', ['xmin', 'ymin'])
    expect(html).toContain('data-testid="viewer-domain-hidden"')
    expect(html).toContain('Hidden patches')
    expect(html).toContain('xmin, ymin')
  })

  it('no line when nothing is hidden', () => {
    expect(render('en')).not.toContain('viewer-domain-hidden')
    expect(render('en', [])).not.toContain('viewer-domain-hidden')
  })

  it('caps the list at eight', () => {
    const html = render('en', ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'])
    expect(html).toContain('n6, n7 +2 more')
    expect(html).toContain('title="n0, n1, n2, n3, n4, n5, n6, n7, n8, n9"')
    expect(html).toContain('n8, n9')
  })

  it('Korean', () => {
    const html = render('ko', ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'])
    expect(html).toContain('숨긴 패치')
    expect(html).toContain('외 2개')
  })
})
