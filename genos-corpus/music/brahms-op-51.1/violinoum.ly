\version "2.12.0"

pizz = ^\markup{\italic{pizz.}}
arco = ^\markup{\italic{arco}}

violinoum= { 
	\relative c' {

		c4. \p \< (d8 ees4. g8 c4. ees8) \!
 		ees2 (fis,4) r8 \crescTextCresc g \< (c4. ees8)
		ees2 (aes,4) r8 a (c4. ees8)
		f2 (a,4) r8 bes (des4. f8)
		aes4. c,8 \! ees4. \< aes8 c4. ees,8 
		aes4. c8 ees4. aes,8 c4. ees8 \!
	%7	
		aes4-. \f r4  g-. \f r4 r2  
		r1 s2
		<bes,,,-. ges'-.>4 \p r4 <a f'>4 r4 r2 
		r1 s2
		f''2 (des c)
		f2 ~ \times 2/3 {f4 \< e f \!} \times 2/3 {bes4 \> f des \!}
		f2 (des c)
		f2 ~ \times 2/3 {f4 \< e f \!} \times 2/3 {d'4 \> b f \!}	%15
		e1. \p 
		(f1.
		e1. 
		ees1. 
		d2) r2 r2
		r1 s2 
		r1 s2
	%22
		r2 r2 r8 aes, g b
		c \< d ees fis g b c d ees g b c \!
		ees-. \f ees,-. ees-. ees'-. ees ees, ees ees' ees ees, ees ees'
		ees ees, ees ees' ees ees, ees ees' \crescTextCresc ees \< ees, ees ees'
		des des, des des' des des, des des' des des, des des'
	%27 
		c ees, ees c' c ees, ees c' c ees, ees c' 
		ces ees, ees ces' ces ees, ees ces' ces ees, ees ces' 
		<ces-. ees,>4 r8 g, bes4. ees8 g4. bes,8
		ees4. ges8 bes4. ees,8 ges4. bes8 
		ees4-. \f r4 d4-. r4 r4 r8 bes
	%32	
		ges'4-. r4 f-. r4 r2
		r1 s2
		r4 bes,-. \p (r4 ges-. r4 ees-.)
		r8 d d ees f2 \< ~ (f8 \> ges \!) ges f 
		r8 ees ees f ges2 \< ~ (ges8 \> bes \!) bes aes
	%37	
		r4 ges \< (bes ees des ges) 
		f (ees des ees des ces) \!
		bes \f (ges) ees (ces) bes' (ges) 
		ees (ces) bes' \> \((f) ees (c)\) \!
		bes8-. \p (aes-.) aes-. d-. d-. ces-. ces-. f-. f-. d-. d-. aes'-.
	%42
		aes d, d ges ges ees ees bes' bes ges ges ees'
		ees4 (fes-.) (r des-. r bes-.)
		r ees-. (r ces-. r aes-.)
		r8  a \p a gis fis2 ~ (fis8 eis) eis fis
		r8 a a gis fis2 ~ (fis8 eis) eis fis
	%47
		r8 \crescTextCresc cis'\< cis b a2 ~ a8 gis gis a \!		
		r8 \< cis cis b b a a gis gis a a gis \!
		gis-. \f a-. a-. gis-. gis a a gis gis a a gis
		gis a a gis gis a a gis gis a a gis
		gis a a gis gis a a gis gis \< a a gis
	%52
		gis a a gis gis a a gis gis a a bes \!
		<< {<bes bes,>2. d,,8 f f bes bes d} \\ \stemUp {d,4 \f} >> \stemNeutral
         	d'8 f f bes bes4. (ees,8)  aes4. (ees8)
		aes4. (d,8) g4. (ees8) ces4. (aes8)
		g4. (bes8) ges4. (bes8) f bes bes d
	%57		
		d f f bes bes4 d,,8 f f bes bes d 
		d f f bes bes4 ees4.-> (c8) aes4 ~
		(aes8 f) d'4. (bes8) g4. (ees8) c'4 ~
		(c8 aes) c4. (aes8) f4. (c8) aes4 ~ 
		(aes8 f) ees4 r2 r2
	%62
		r2 r2 c' \p 
		(bes2. a4 bes) ees
		(c bes2 a4 bes4) ees ~
		(ees8 d ees a,) ~ a bes (ees g ~ g fis \< g) d ~
		d ees (a c ~ c) a (bes ees ~ ees cis d) g ~ \!
	%67
		g e (f d ees \> c aes ees c aes ees c) \! 
		b \p (c d c ees aes c ees) d (ees f ees 
		bes' g aes f ges ees c a ges ees d c)
		bes (a gis a c ees ges c) bes (a gis a
		ees' g, ees g) f' (ees d ees g ees g, ees)
	%72
		f (ees d ees ees' g, ees) g (f' ees d ees 
		g ees g, ees) f \< (ees d ees bes'2) ~ \!
 		bes4 \> (aes g2 \! f2) 
		ees r2 r2 
		r1 s2
		r2 r2 bes'\< ~ 
		(bes4 \! \>  a g2 \! f)
	%79
		ees r2 r2 
		r1 s2
		r2 r g ~ 
		(g4 f ees2 d)
		r2 r aes' \p ~
		aes2. (b4 c2 ~ 
		c4 d e2. gis4
		a2. b4 c2) ~ \key c \major
	%84	
		c8 \p (a c a \repeat "tremolo" 2 {c8 a8} \repeat "tremolo" 2 {c8 a8)} 
	 	\repeat "tremolo" 2 {c8 (a8} \repeat "tremolo" 2 {c8 a8} \repeat "tremolo" 2 {c8 a8)} 
		\repeat "tremolo" 2 {d8 (bes8)} \repeat "tremolo" 2 {bes8 (f8)} \repeat "tremolo" 2 {f8 (d8)}
		\repeat "tremolo" 2 {f8 (d8} \repeat "tremolo" 2 {e8 c8} \repeat "tremolo" 2 {d8 b8)}
		a4. \< (b8 c4. e8 a4. c8) \!
		\crescTextCresc c2 \< (ees,4) r4 r2
	%90	       
		bes4. (c8 d4. f8 bes4. d8) 
		f2-. e-. <gis,-. b, e,> 
		<a-. c, e,>4 \! \f r e4. c8 a4 r4     
	    %adicionar o marc.
		a'4. b8 c4 r a,4. b8 
		c4 r4 r2 r2 
		<g'-. c, e,>4. e8 c2 << {<b f'>2} \\ \stemUp {d,4} >> \stemNeutral
	%96
		r e'4-. \p (r des-. r bes-.)
		r a'4-. (r f-. r e-.)
		r8 dis dis  \< e fis2 \! \> ~ (fis8 c) \! c  b
		r8 a a  \< b c2 \! \> ~ (c8 a) \! a  fis
		r2 b4. \f g8 e4 r4
	     %adicionar o marc.
	%101
		g'1  \f a2 
		b4 r r2 r
		d4. b8 g2 c
		r4 b4-. \p (r aes-. r f-.)
		r e'4-. (r c-. r b-.) 
		r8 ais ais  \< b cis2 \! \> ~ (cis8 g) \! g  fis
	%107		
		r8 e e  \< fis g2 \! \> ~ (g8 e) \! e  cis
		R1*3/2
		r8 f f  \< g aes2 \! \> ~ (aes8 f) \! f  d
		R1*3/2
	%111
		r8 fis fis  \< gis a2 \! \> ~ (a8 fis) \! fis  dis \key e \major
		R1*3/2	
		a4. \p (\crescTextCresc b8 \< cis4. e8 a4. cis8)
		r8 fis \f fis   gis a2 ~ (a8 fis)  fis  d 
	%115		
		r8 bis bis cis dis2   ~ (dis8 a)  a  gis 
		r8 fis fis gis a2   ~ (a8 fis)  fis  d
 		r8 bis bis cis dis2   ~ (dis8 a)  a  gis 
		r8 cis \f  cis bis r8 gis' gis fis fis d d cis
	%119 
		r8 fis fis eis r8 cis' cis b b g g fis
		r8 a a gis r8 fis fis e r8 dis dis cis
		cis d d fis fis cis cis e e bis bis dis
		fis4. \f gis8 a2 r2
		b4. cis8 d2 r2
	%124
		dis4. e8 eis4. fis8 fisis4. gis8
		gis2 ~ gis8 a a gis gis2 ~
		gis8 a  a gis gis2 ~ gis8 a a gis
		\crescTextCresc gis4. \< (a8) a4. (gis8) gis4. (a8)
		a4. \< (gis8) gis4. (b8) b4. (a8) 
	%129
		a8-. \f fis-. fis-. e-. r4 a,,8 fis fis e r4
		a''8 e e d r4 a,8 e e d r4 
		a'''8 d, d cis r4 a,8 d, d cis r4 
		a'''8 cis, cis bis r4 a,8 cis, cis bis r4 \key ees \major
	%133
		r2 c4. \p (d8 ees2)
		r2 c4. (d8 ees2)
		r2 c1 
		(d1 ees2) ~
		ees2 ~ ees4.  \< (aes8 c4. ees8 \!)
		\crescTextCresc ees2 \< (fis,4) r8 g (c4. ees8)
		ees2 (aes,4) r8 a8 (c4. ees8)
		
	%140	
		f2 (a,4) r8 bes8 (des4. f8)
		aes4. c,8 ees4. aes8 c4. ees,8
		aes4. c8 ees4. aes,8 c4. ees8
		aes4-. \f r4 ees-. r4 r2
		R1*3/2
		<g,,-. b,>4 \p r4 <fis-. ais,>4 r4 r2	    	
		R1*3/2
		e4-. \pizz r4 d-. r4 r2 		
	   	R1*3/2
	%149
		r4 fis r4 g r4 a
		r4 g r4 ees r4 d
	   	g'2 \arco (ees d)
	   %colocar espress. 			
		g ~ (\times 2/3 {g4 \< fis g \! } \times 2/3 {c4 \> g ees \!)} 
		g2 (ees d)
	   %colocar arco e espress. 			
		f ~ (\times 2/3 {f4 \< e f \! } \times 2/3 {d'4 \> b f \!)}
		e1. \p (
	%156		
		f
		e
		\dimTextDim e,4. \> (f8 g2 e)
		f2  \> (a,1 \!)
		R1*3/2
		R1*3/2
		g'2 \p (\crescTextCresc e \< des)	
		c1 ~ (c8 des c e \<)
	%161
		f g aes b c e f g aes b c f 
		d \f d, d d' d d, d d' c c, c c'
		ces ces, ces ces' ces ces, ces ces' bes \crescTextCresc bes, \< bes bes'
		bes bes, bes bes' bes bes, bes bes' ees ees, ees ees'
		des des, des des' des des, des des' des des, des des'
	%169
		des des, des des' des des, des des' des des, des des'
		<des des,>4 r8 c,,8 ees4. aes8 c4. ees,8
		aes4. c8 ees4. aes,8 c4. ees8
		aes4-. r4 g-. r4 r4 r8 ees8
		c'4-. r4 b4-. r4 r2
		R1*3/2
	%175    
		r4 g4-. \p ( r4 ees-. r4 c-.)
		r8 b b \< c d2 \> ~ (d8 ees \!) ees d
		r8 c c \< d ees2 \> ~ (ees8 g \!) g f
		r4 ees (g c \< bes ees)
		d (c bes c bes aes)
	%180
		g \f (ees c aes) g'  
		(ees c aes)  g' \> (d c a)
		g8 \p (f)  f-. b-. b aes aes d d b b f'
		f b, b ees ees c c g' g ees ees c'
		c4 (des-.) (r bes-. r g-.)
	%185
		r c-. (r aes-. r f-.)
		r8 ges \p ges f ees2 ~ (ees8 d) d ees
		r8 ges ges f ees2 ~ (ees8 d) d ees
		r8 \crescTextCresc bes' \< bes aes ges2 ~ (ges8 f) f ges
		r8 bes \< bes aes aes ges ges f f ges ges f \!
	%190
		eis-. \f fis-. fis-. eis-. eis fis fis eis eis fis fis eis
		eis fis fis eis eis fis fis eis eis fis fis eis
		eis fis fis eis eis fis fis eis eis fis fis eis
		eis \< fis fis eis eis fis fis eis eis fis fis g \!
		<< {<g b,>2. b,,8 d d g g b} \\ \stemUp {d,4 \f s4 s1} >> \stemNeutral
	     %atenção na linha acima
	%195
		b'8 d d g g4. (c,8) f4. (c8)
		f4. (b,8) e4. (c8) aes4. (f8)
		e4. (g8) ees4. (g8) d g g b
		b d d g g4 b,,8 d d g g b
		b d d g g4 c4.-> (a8) f4 ~ 
	%200
		(f8 d) b'4. (g8) e4. (c8) a'4 ~
		(a8 f) a4. (f8) d4. (a8) f4 ~
		(f8 d) c4 r2 r2
		r2 r2 a'2 \p 
	    %colocar o dolce
		(g2. fis4 g) c ~ 
	%205	
		(c b c fis, g) c ~
		c8 (b c fis, ~ fis g c e ~ e dis e) b ~
		b (a fis' a ~ a \< fis g c ~ c ais b e) ~
		e (cis d b  c \p a f c a f d a)
		gis (a b a c f a c b c d c)
	%210
		g' (e f d ees d a fis ees c b a)
		c ( ees fis a c ees fis a) g (fis eis fis
		c' e, c e) d' (c \< b c e c e, c \!)
		d (c b d c' e, c e) d' (c b c
		e c e, c) d (c b c g'2) ~ 
	%215
		g4 \> (f e2 \! d)
		c2 r2 r2
		R1*3/2
		r2 r2 g'2 \< ~
		(g4 \> f \! e2 d2)
		c2 r2 r2
		R1*3/2
		r2 r2 g2 \p ~
 	   %adicionar o piú		
		(g4 f ees2 d) \time 2/2
	%224
		r4 \p des4-. (r4 bes4-.)
	   %adicionar crescendo ed agitato
		r4 aes'-. (r4 f4-.)
		r8 b b c d2 ~
		d8 \< ees ees f f g aes g
		r4 des4-. \p (\crescTextCresc r4 \< bes4-.)
		r4 aes'4-. (r4 f4-.)
		r8 b b c d2 ~
	%231 
		d8 \< ees ees f f g aes g
		g2 \f g,
		c,4. d8 e g c e
		<a a,,>2 <c, a,>2
		f,4. g8 a c f a
		c4-. c \f (des c
		bes aes f des)
	%238
	   	r4 c2 (ees,4)
		r4 aes'2 (b,4)
		r4 c (des c
		bes aes f des)
		r4 c2 (ees,4) 
		r4 aes'2 (b,4)
	%244 ao 260 - À partir daqui cada linha representa os acordes ligados e estão escritos em "duas vozes".		
		r2 << {<c e,>2 ~ <c e,>4 b4 \rest 
		<c f,>2 ~ <c f,>4 b4 \rest 
		<des f,>2 ~ <des f,>4. ees8 \< f a des f \! e4 b,4 \rest
		<c e,>2 ~ <c e,>4 b4 \rest <c f,>2 ~ <c f,>4 b4 \rest 
		<des f,>2 ~ <des f,>4. ees8 \< f a des f \! e4 b,4 \rest 
		<c' e,>2 ~ \dimTextDim <c e,>4 \> b,4 \rest \!
		<g' c,>2 ~ <g c,>4 b,4 \rest 
		<c e,>2 ~ <c e,>4 b4 \rest 
		<g g,>2 ~ <g g,>4 b4 \rest 
		<e, g,>2 ~  <e g,>4 b'4 \rest 
		g,2 ~ g4 b'4 \rest  
		g,2 ~ g4 b'4 \rest  b2 \rest 
		<< <c c'>1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >> }
		\\ \stemUp { g,4 \fz s4 s2 
		a4 \fz s4 s2 
		aes4 \fz s4 s1 s2 
		g4 \fz s4 s2 
		a4 \fz s4 s2 
		aes4 \fz s4 s1 s2 
		g'4 \fz s4 s2 
		e4 s4 s2 
		g,4 s4 } >>
		

 }	
}
