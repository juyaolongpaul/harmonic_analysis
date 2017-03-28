\version "2.12.0"
\header {
  title = "Sonata Op 2, III, Trio"
  composer = "Beethoven"
}
\new PianoStaff <<
  \set Score.currentBarNumber = #13
  \bar ""
  \new Staff {
    \clef treble
    \time 3/4
    \relative c'' {
      f4( e d
      c! b\< a
      gis^\markup{ 7 }
      f^\markup{ 6 }
      e
      d c b)\!
      <c c'>\fp( b' a)
      <<
        { 
          f'4\sf( e d) 
        }
        \\
        { 
          gis,2. 
        }
      >>
    }
  }
  \new Staff  {
    \clef treble
    \relative c'{
      d8 a' c,! a' b, gis'
      \clef bass
      a, a' g,! g'! f, f'
      e, e' d, d' c, c'
      b, b' a, a' gis, gis'
      <a, a'> e'' b e c e
      <<
        { 
          d8 e c e b e 
        }
        \\
        { 
          e,2.\sf 
        }
      >>
    }
  }
>>