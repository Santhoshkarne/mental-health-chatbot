import { create } from 'zustand'
import { messageContentSchema } from '@/schemas/chat'
import type { ChatMessage, ChatSummary, UserProfile } from '@/types/chat'

const makeId = () => Math.random().toString(36).slice(2, 10)

const seedChats: ChatSummary[] = [
  { id: makeId(), title: 'Summarizer implementation vs RAG', updatedAt: Date.now() - 1000 * 60 * 5, messages: [] },
  { id: makeId(), title: 'github.com/SaisrikarVo…', updatedAt: Date.now() - 1000 * 60 * 20, messages: [] },
  { id: makeId(), title: 'Simplify project tech stack display', updatedAt: Date.now() - 1000 * 60 * 60, messages: [] },
  { id: makeId(), title: 'Coding assessment integrity and…', updatedAt: Date.now() - 1000 * 60 * 90, messages: [] },
  { id: makeId(), title: 'Splitkaro architecture documentation', updatedAt: Date.now() - 1000 * 60 * 120, messages: [] },
  { id: makeId(), title: 'Splitkaro code quality and architecture', updatedAt: Date.now() - 1000 * 60 * 150, messages: [] },
  { id: makeId(), title: 'LaTeX resume generation from repo', updatedAt: Date.now() - 1000 * 60 * 200, messages: [] },
  { id: makeId(), title: "Dijkstra's algorithm time complexity", updatedAt: Date.now() - 1000 * 60 * 240, messages: [] },
]

interface ChatState {
  sidebarCollapsed: boolean
  chats: ChatSummary[]
  activeChatId: string | null
  user: UserProfile

  toggleSidebar: () => void
  createChat: () => void
  selectChat: (id: string) => void
  sendMessage: (content: string) => { success: boolean; error?: string }
}

export const useChatStore = create<ChatState>((set, get) => ({
  sidebarCollapsed: false,
  chats: seedChats,
  activeChatId: null,
  user: { name: 'srikar', plan: 'Free plan', initial: 'S' },

  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),

  createChat: () => set({ activeChatId: null }),

  selectChat: (id) => set({ activeChatId: id }),

  sendMessage: (content) => {
    const parsed = messageContentSchema.safeParse(content)
    if (!parsed.success) {
      return { success: false, error: parsed.error.issues[0]?.message ?? 'Invalid message' }
    }
    const trimmed = parsed.data

    const { activeChatId, chats } = get()
    const userMsg: ChatMessage = {
      id: makeId(),
      role: 'user',
      content: trimmed,
      createdAt: Date.now(),
    }
    const assistantMsg: ChatMessage = {
      id: makeId(),
      role: 'assistant',
      content: "I'm MindMend, here to help. This is a demo reply so you can see how the conversation view looks.",
      createdAt: Date.now() + 1,
    }

    if (activeChatId) {
      set({
        chats: chats.map((c) =>
          c.id === activeChatId
            ? { ...c, messages: [...c.messages, userMsg, assistantMsg], updatedAt: Date.now() }
            : c,
        ),
      })
      return { success: true }
    }

    const newChat: ChatSummary = {
      id: makeId(),
      title: trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed,
      updatedAt: Date.now(),
      messages: [userMsg, assistantMsg],
    }
    set({ chats: [newChat, ...chats], activeChatId: newChat.id })
    return { success: true }
  },
}))
