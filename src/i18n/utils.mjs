import {
  defaultHtmlLang,
  defaultLocale,
  localeHtmlLangs,
  locales,
  routeLocales,
} from './config.mjs';

export function isSupportedLocale(locale) {
  return locales.includes(locale);
}

export function getHtmlLang(locale) {
  return localeHtmlLangs[locale] ?? defaultHtmlLang;
}

export function normalizeLocale(locale) {
  return isSupportedLocale(locale) ? locale : defaultLocale;
}

export function stripLocaleFromPath(pathname) {
  const normalizedPath = normalizePath(pathname);
  const parts = normalizedPath.split('/').filter(Boolean);
  if (parts.length === 0) return '/';

  const maybeLocale = parts[0];
  if (!isSupportedLocale(maybeLocale) || maybeLocale === defaultLocale) {
    return normalizedPath;
  }

  const rest = parts.slice(1).join('/');
  return rest ? `/${rest}` : '/';
}

export function localizePath(pathname, locale) {
  const targetLocale = normalizeLocale(locale);
  const basePath = stripLocaleFromPath(pathname);

  if (targetLocale === defaultLocale) return basePath;
  return basePath === '/' ? `/${targetLocale}` : `/${targetLocale}${basePath}`;
}

export function normalizePath(pathname) {
  const pathOnly = String(pathname || '/').split('?')[0].split('#')[0];
  const withLeadingSlash = pathOnly.startsWith('/') ? pathOnly : `/${pathOnly}`;
  if (withLeadingSlash.length > 1) return withLeadingSlash.replace(/\/+$/, '');
  return '/';
}

export function getPostBaseSlug(post) {
  return stripLocaleFromSlug(post.slug);
}

export function stripLocaleFromSlug(slug) {
  const parts = String(slug).split('/').filter(Boolean);
  if (parts.length > 1 && routeLocales.includes(parts[0])) {
    return parts.slice(1).join('/');
  }
  return parts.join('/');
}

export function isDefaultLocalePost(post) {
  const firstPart = String(post.slug).split('/').filter(Boolean)[0];
  return !routeLocales.includes(firstPart);
}

export function getDisplayDate(post) {
  return post.data.updatedDate ?? post.data.pubDate;
}

export function sortPostsByDisplayDate(posts) {
  return [...posts].sort((a, b) => {
    const dateDiff = getDisplayDate(b).valueOf() - getDisplayDate(a).valueOf();
    if (dateDiff !== 0) return dateDiff;
    return getPostBaseSlug(a).localeCompare(getPostBaseSlug(b));
  });
}

export function getLocalizedPost(basePost, locale, posts) {
  const targetLocale = normalizeLocale(locale);
  if (targetLocale === defaultLocale) return basePost;

  const baseSlug = getPostBaseSlug(basePost);
  return posts.find(post => post.slug === `${targetLocale}/${baseSlug}`) ?? basePost;
}

export function getAvailablePostLocales(basePost, posts) {
  const baseSlug = getPostBaseSlug(basePost);
  return [
    defaultLocale,
    ...routeLocales.filter(locale => posts.some(post => post.slug === `${locale}/${baseSlug}`)),
  ];
}

export function getLocalizedPostList(posts, locale) {
  const basePosts = posts.filter(isDefaultLocalePost);
  return basePosts.map(post => getLocalizedPost(post, locale, posts));
}
