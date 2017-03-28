\version "2.12.0"

\header {
  title = "An die Freude, k. 53"
  composer = "Mozart"
}

\new PianoStaff <<
  \new Staff {
    \relative c'{
      \key f \major 
      \time 2/4
      f4.( g16 a)
      g2
      a8 g16( a) bes8 g
      e f r4
      f4. a16( c )
      bes4. g16( f)
      e4. g16( bes)
      a4 r4
      c4. a16( f)
      d'4 c 
      c8( bes ) bes( a)
      a g r4
      g4. c8
      a4. f'16( d)              
      c4. b8
      c4 r4 
      c4. e16( c)
      a8( f') e( d)
      c4 b
      c r4
      a4. bes16( c)
      bes4 a8( g)
      ees'4 d8( c)
      bes16( a) g8 r4
      c4. d16( e?)
      f8( e) d( c)
      bes( a) g( f)
      c'4 r4
      c4. d16( bes) 
      a8( f') \times 2/3 {e8( f d)}
      c4. d16( bes) 
      bes8 a r4
      c4. a16( f) 
      d'4. c16( bes)
      a8( c) e,( g)
      f4 r
      c'4. a16( g)
      d'16( f e g f8) bes,
      a4 g4
      f4 r4
      \bar "|."
    }
  }
  \new Staff { 
    \relative c'{
      \key f \major 
      \time 2/4 
      \clef bass
      r8 a( g f)
      r e( d c)
      f f e bes
      c d r4
      r8 a' a a
      r g g g 
      c, c c c
      f f g g
      a a a a
      bes bes, a a' 
      g e f b,
      c c d d
      e e e e
      f f f f
      g g g g
      a a g f
      e e e e
      f f f f
      g g g, g
      c c'16 d c bes a g
      fis8 fis fis fis
      g g g g
      fis fis fis fis
      g g f f
      e e bes' bes
      a c bes a
      bes a g f
      c c c c
      e e e e
      f f f f
      e e e e
      f f f f
      a a a a
      bes bes bes bes
      c c c, c
      d d d d
      a a a a
      bes bes' a bes
      c c c, c
      f4 r
    }
  }
>>