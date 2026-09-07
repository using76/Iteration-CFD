import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'

const remarkPlugins = [remarkGfm]
const rehypePlugins = [rehypeHighlight]

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={{ a: (p) => <a {...p} target="_blank" rel="noreferrer" /> }}>
      {text}
    </ReactMarkdown>
  )
})
