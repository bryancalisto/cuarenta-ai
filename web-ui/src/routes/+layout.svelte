<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<title>Cuarenta Arena</title>
	<meta
		name="description"
		content="A multiplayer Cuarenta lobby for signing in, creating games, joining by code, and syncing live state."
	/>
	<link rel="icon" href={favicon} />
</svelte:head>
{@render children()}
