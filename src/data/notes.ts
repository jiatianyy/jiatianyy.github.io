import { getCollection, type CollectionEntry } from 'astro:content';
export const isReview = import.meta.env.DEV || import.meta.env.MODE === 'review';
export const visibleNote = (note: CollectionEntry<'notes'>) => isReview || (!note.data.draft && note.data.approved && note.data.kind === 'research');
export async function getNotes() {
  return (await getCollection('notes', visibleNote)).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}
