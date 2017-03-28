\version "2.12.0"

pizz = ^\markup{\italic{pizz.}}
arco = ^\markup{\italic{arco}}

viola= { 
	\clef alto 		
	\relative c {
	 	
		ees8 \p ees8 ees ees ees2:8 ees 2:8
		ees2:8 ees2:8 \crescTextCresc ees2:8 \< 
		ees2:8 ees2:8 ees2:8
		ees2:8 ees2:8 des2:8
		c2:8 c2:8 \< c2:8
		<c c'>2:8 <c c'>2:8 <c c'>2:8 \!
	%7
		
		<c'-. ees>4 \f r <g g'>1 \fp ~ 
		<g g'>1. 
		des'4-. \p r <f f,>1 ~ 
		<f f,>1. 
		f,4. \p (g8 aes2 f4) r
		bes4. (des8 \< f2 \> f,4) \! r
		f4. (g8 aes2 f4) r
		b4. \< (d8  f2 \> f,4) \! r
	%15
		g4. (b8 c2) r2
		r2 aes (c)
		g4. (b8 c2) r2
		r2 a (c)		
		g4 r4 r2 r2
		\dimTextDim b \> (g fis)
		r2 fis r2
	%22
		g \< (ees d) \!
		c4. \< (d8 ees4. g8 c4. ees8) \!
 		ees2 \f (fis,4) r8 g (c4. ees8)
		ees2 (aes,4) r8 \crescTextCresc a \< (c4. ees8)
		f2 (a,4) r8 bes (des4. f8)
	%27		
		aes4. c8-. \sf aes4. ees8-. c4. aes'8-. \sf 
		ees4. ces8-. aes4. ees'8-. \sf  ces4. aes8-.
		ees4-. r <bes' g'>2:8 <bes g'>2:8 
		<bes ges'>2:8 <bes ges'>2:8 <bes ees,>2:8 
		<a-. ges'>4 \f r <bes-. f'>4 r r2
	%32
		<a-. a'>4 r  bes8-. \fp bes'-. bes-. bes,8-. bes8-. bes'-. bes-. bes,8-.
		bes8 aes' aes bes,8 bes8 f' f bes,8 bes8 d d bes8 
		bes8 ees ees bes8 bes8 ges' ges bes,8 bes8 a' a bes,8 
		bes8 bes' bes bes,8 bes8 bes' bes bes,8 bes8 bes' bes bes,8
		bes8 bes' bes bes,8 bes8 bes' bes bes,8 bes8 bes' bes bes,8
	%37
		ges' bes bes ges ees \< ges es ees ges des des ges
		aes,8 aes' aes aes,8 aes8 aes' aes aes,8 aes8 aes' aes aes,8 \!
		ges'8 ges, ges ges'8 ges8 ges, ges ges'8 ges8 ges, ges ges'8
		ges8 ges, ges ges'8 f8 \> f, f f'8 f8 f, f f'8 \!
		bes,4 \p f'-. r f-. r ces'-.
	%42
		r bes-. (r bes-. r ges-.)
		r g-. (r g-. r bes-.)
		r8 g,-. g-. ces-. ces-. aes-. aes-. ees'-. ees-. ces ces aes' 
		c,1 \p cis 2
		d1 dis 2  
	%47		
		\crescTextCresc e \< eis fis
		g \< e g, \!
		r4 fis \f (a d cis fis)
		e (a,) r f (a d
	%51 e 52	
		cis ) << {a'2 a a4 ~ a a2 a2 s4  } \\ {f e d cis f (e) d (cis) f (e) a,} >>  
	%53   
	        d,8 \f f f bes <bes f'>1 ~
	    	<bes f'>2  ees4 (c) ees4 (c)	
		d4 (ces) ees4 (bes) ees4 (ces)
		ees4. (bes8)  a c c ees, bes' d d  f 
	%57
		f bes bes d d2. \clef treble d,8 f
		f bes bes d ees4.-> (c8) aes4.-> (f8)
		d'4.-> (bes8) g4.-> (ees8) c'4.-> (aes8) \clef alto
		c4.-> (aes8) f4. (c8)  aes4. (f8)
		<f aes>4 \dimTextDim <f-. aes>4 \> (r <f-. aes>4 r <f-. aes>4)  
	%62
		r <f a>4 r <f a>4 \! r <f c'>4 \p	  
		r <g bes>4 r <f a>4 r <g bes>4
		r <f c'>4 r  <f a>4  r <g bes>4 
		r <f a>4 r <g bes>4 r a	\<	
		r c r bes r bes \!
	%67
		<aes-. aes'>2  (<aes-. aes'>2) \p  r2
		R1*3/2
		<bes-. f'>2  (<a-. c>2) r2
		r2 r2 ees' ~ 		
		ees ees,1 \< 
	%72
		ees \! \< ees2-. \p 
		(r2 f-.) r2
		aes1.
		g2 \p \<(ges1) \! ~
		ges1. \> 
		f1. \! ~
		f2 (bes aes)
	%79
		g (fis1 \<
		a1.) \! 
		(g1. ~ g1.)
 		a1. \p
	     %ligar semibreve com nota anterior
		<< {c1. _~ c1. } \\ 
		{ees1. ^(e1.)
		 ^~ e1. } >>  \key c \major
	%84	
		a,4. \p (b8 \< c4. e8 a4. c8 \!)
		c2 \> (ees,4 \!) r r2
		bes4. \p (c8 \< d4. f8 bes4. d8 \!)
		d2 \> (gis,4 \!) r r2 
		\repeat "tremolo" 2 {a,8 \< (c8} \repeat "tremolo" 2 {a8 c8} \repeat "tremolo" 2 {a8 c8 \!)} 
		\crescTextCresc \repeat "tremolo" 2 {f,8 \< (c'8} \repeat "tremolo" 2 {f,8 c'8} \repeat "tremolo" 2 {f,8 c'8 )} 
	%90
		\repeat "tremolo" 2 {f8 (d8} \repeat "tremolo" 2 {f8 d8} \repeat "tremolo" 2 {f8 d8)} 	
		\repeat "tremolo" 2 {b'8 (gis8} \repeat "tremolo" 2 {b8 e,8)} e2-.
		a,4. \f b8 c4 r a'4. c8
	    %colocar o marc.
		c4 r c,4. a8 f4 r
		c'4. c8  a2 b
		c4 r r2 r2
	%96
		c,8-. \p c'-. c-. c,-. c8-. c'-. c-. c,-. c8-. c'-. c-. c,-. 
	        c8 c' c c,  c8 c' c c,    c8 c' c c,
		b' c c b b c c b b dis dis c
		b c c b b e e b b dis dis b 
		r2 r e4. \f gis8
	%101
		g4 r b,4. g8 e4 r
		<d g d'>4. b'8 g2 c
		b4 r r2 r2 
		g8 \p g' g g, g g' g g, g g' g g, 
		g g' g g, g g' g g, g g' g g, 
		fis fis' fis fis, fis fis' fis fis, fis fis' fis fis,
	%107
		fis fis' fis fis, fis fis' fis fis, fis fis' fis fis,
		fis fis' fis fis, fis fis' fis fis, fis fis' fis g
		g, g' g g, g g' g g, g g' g g, 	
  		g g' g g, g g' g g, g g' g gis 	
	%111		
		gis, gis' gis gis, gis gis' gis gis, gis gis' gis gis, \key e \major
		gis \p \crescTextCresc gis' \< gis gis, gis gis' gis gis, gis gis' gis gis,
		a e' e e, e e' e e, e e' e a,
		a \f a' a a, a a' a fis fis d' d f, 
	%115
		fis gis gis fis fis gis gis bis, bis dis dis gis, 
		a a' a a, a a' a fis fis dis' dis fis, 
		fis gis gis fis fis gis gis bis, bis dis dis gis, 
		fis4. \f gis8 a2 r2
	%119
		b4. cis8 d2 r2
		fis4. e8 dis4. cis8 fis4. e8
		a4-. r4 cis,-. r4 gis4-. r4
		r8 cis-. \f cis bis r gis' gis fis fis d d cis
		r  fis fis eis e cis' cis b b g g fis
	%124
		r a a gis r b b a r cis cis b
		r d d cis cis2 ~ cis8 d d cis 
		cis2 ~ cis8 ~ d d cis cis2 ~ 
		\crescTextCresc cis8 \< (d) (d4 ~ d8 e,) (e4 ~ e8 d) d4 ~
		(d8 e,) e4 ~ (e8 d) d4 ~ (d8 e e4)
	%129
		< cis a'>4 \f r8 a''8-. fis-. e-. a,-. fis-. fis-. e-. cis-. cis-.  
	  %colocar o marc. acima
		fis4-. r8 a'8 e d a e e d fis fis
		e4-. r8 a'8 d, cis a d, d cis e e
		dis4-. r8 a''8 cis, bis a cis, cis bis cis cis \key ees \major
	%133
		c2:8 \p c2:8 c2:8
		c2:8 c2:8 c2:8
		c2:8 c2:8 c2:8
		c2:8 c2:8 c2:8
		c2:8 c2:8 \< <ees c'>2:8
		\crescTextCresc <ees c'>2:8 \<  <ees c'>2:8  <ees c'>2:8
		 <ees c'>2:8  <ees c'>2:8  <ees c'>2:8  
	%140
		<f c'>2:8  <f c'>2:8  <f bes>2:8
		<ees c'>2:8  <aes c>2:8  <aes c>2:8  
		<aes c>2:8 <aes c>2:8 <aes c>2:8 
		<ees c'>4-. \f r4 <g g'>1 \fp ~ 
		<g g'>1.
		d'4-. \p r4 <gis gis,>1 ~ 
		<gis gis,>1.
		e4-. r4 d1 ~ 
		d1.
	%149
		d2 (cis c
		bes aes fis)
		g4. (aes8 g1) ~
	    %adicionar arco.
		g4. (b8 c2 g) ~
		g4. (aes8 g1) ~
		g4. (a8 b2 g)
		g4. (b8 c1) ~
	%156
		c4 (g' aes f c f,)
		g4. (b8 c2) r2
		\dimTextDim  c4. \> (d8 e2 c)
		c,2. \> (d4 e) \! r4
		a2 \> (f e) \!
		r2 e2-_ r2
	     %colocar tenuto mais staccato
	 	g2 \p (\crescTextCresc e \< des)	
		<c c'>1. 
	%164
		f4. \< (g8 aes4. c8 f4. aes8) \!
		aes2 \f (b,4) r8 c8 (f4. aes8)
		aes2 (des,4) r8 d8 (\crescTextCresc f4. \< aes8)
		bes2 (d,4) r8 e8 (ges4. aes8)
		des4. f8-. \sf des4. aes8 f4. des'8-. \sf 
	%169
		aes4. fes8 des4. a'8-. \sf fes4. des8
		aes4-. r4 <ees c'>2:8  <ees c'>2:8
		<ees c'>2:8  <ees c'>2:8 <ees c'>2:8
		<ees c'>4-. r4 <aes g'>4-. r4 r4 r8 c8
		<c aes'>4-. r4 g8-. \fp g'-. g-. g,-. g-. g'-. g-. g,-.
		g g'g g,  g g'g g, g g'g g,
	%175
		g g'g g,  g g'g g, g g'g g,
		g g'g g,  g g'g g, g g'g g,
		g g'g g,  g g'g g, g g'g g,	
		ees' g g ees c ees ees c ees bes bes ees
		f, f'f f,  f f'f f, f f'f f,
	%180
		ees' \f ees, ees ees' ees ees, ees ees' ees ees, ees ees' 	
		ees ees, ees ees' d \>  d, d d' d d, d d'
		g,4 \p d'-. (r4 d-. r4 aes'-.)
		r4 (g-. r g-. r ees-.)
		r4 (e-. r e-. r g-.)		
	%185
		r8 e,-. e-. aes aes f f c' c aes aes f'
		a,1 \p  bes2 
		b1 c2
		\crescTextCresc des2 \< d e
		e \< cis e, \!
	%190
		r4 dis \f (gis b ais dis)
		cis ( fis) r4 d, (fis b 
		ais) d ( cis) b ( ais) d
		(cis) \< b (ais) d ( cis fis,)
		b8 \f ees, ees g <g d'>1 ~
	%195	
		<g d'>2 c4 (a) c (a)
		b (aes) c ( g) c (aes)
		c4. (g8) fis a a c, g' b b d 
		d g g b b2 b,8 d d g
		g b b4. (a8) e4. (d8) f4 ~
	%200
		(f8 d) d4. (c8) e4. (c8) a'4 ~
		(a8 f) a4. (f8) d4. (a8) f'4 ~
		(f8 d)\dimTextDim c4. \> (d8) c4. \> (d8) c4 ~
		(c8 d) c4. (d8) c4. (d8) c4 \p ~ 
	    %colocar p dolce
		(c8 e) c4. (d8) c4. (e8) c4  ~
	%205
		(c8 d) c4. (d8) c4. (e8) c4  ~
		(c8 d) c4. (e8) c4. (e8) c4  ~
		(c8 fis) c4. \< (e8) b4. (g'8) b,4 
		b2-. \p (c-.) r2 
		r2 r2 f2
	%210
		( <d g,>2-.)  (< c a >2-.) r2 
		r2 r2 a 
		(g) fis \< (g)
		a (g) fis-. \p 
		(r2 g-.) r2
	%215
		f'1. \p 
		e2 (ees1) ~ \<
		<< ees1. ({ s4 \> s4  s4\! s4} >>
		e1.)
		(d2) (f,1)
		<< {g2 \p ~ g1 \< } \\ \stemUp {e2 (ees1)} >>
	  % acima colocar o piú p
	  	<< { << a1. ({ s4 \> s4  s4\! s4} >> g1.) ~ g} \\ {ees1. ~ ees (d)} >>  \time 2/2
	 %224
		r4 \p \crescTextCresc e4-. \< (r4 e4-.)
		r4 f4-. (r4 aes4-.)
		r4 aes'-> ~ aes8 f f c
		b4-. \< aes2-> g4-.
		c,8 \p \crescTextCresc c' \< c c, c c' c c,
		c c' c c,  c c' c c,  
		c c' c c,  c c' c c,  
	%231
		c \< c' c c,  c c' c c,
		c4. \f d8 e g c e 
		<e g,>1:8
		f,4. g8 a c f a
		<f a>1:8
		<f aes>1:8 \f
		<f aes>1:8
	%238
		<ees c'>1:8
		<d f>1:8
		<c aes'>1:8
		<des f>1:8
		<g, ees'>2:8 c2:8
		g1:8 
		<c, c'>2:8 <c c'>2 \fz ~ 
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <c c'>2 \fz ~
	%246
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <des' f>2 \fz ~
		<des f>8 <des f>8-. <des f>8-. <des f>8-. <f, aes>2:8
		<c c'>2:8 <c c'>2 \fz ~ 
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <c c'>2 \fz ~
		<c c'>8 <c c'>8-. <c c'>8-. <c c'>8-. <des' f>2 \fz ~
		<des f>8 <des f>8-. <des f>8-. <des f>8-. <f, aes>2:8
	%252
		<g g'>1:8 
		\dimTextDim <g g'>1:8 \>
		\times 2/3 {g4-. g-. g-.} \times 2/3 {g4-. g-. g-.}
		\times 2/3 {g4-. g-. g-.} \times 2/3 {g4-. g-. g-.}
		e4-. e4-. e4-. e4-.
		e4-. e4-. e4-. e4-.
		e4-. r4 r2
		e4-. r4 r2
		<< <e c'>1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >>
		
	 

  }
}
