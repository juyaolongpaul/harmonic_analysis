\version "2.12.0"
\header {
  title = "Well-Tempered Clavier, Book I, Prelude 10"
  composer = "Bach"
}
\new PianoStaff <<
  \set Score.currentBarNumber = #40
  \bar ""
  \new Staff {
    \clef treble
    \time 4/4
    \relative c'' {
      <<
        {
          e16 d! c b a g f e c' b a g f e dis cis
          dis[ b f'8 ] r16 e8 dis16 <gis, b e>2 \bar "||"
        }
        \\
        {
          g'8 r8 s4 s2
          s16 b,8. ~ b8[ a] s2
        }
      >>
    }
  }
  \new Staff {
    \clef bass
    \relative c {
      <<
        {
          s2 a16 b 
          c^\markup{ 6 }
          d^\markup{ 7 }
          e f g a ~
          a8. a16 g8[ f] <e, e'>2
        }
        \\
        {
          <c' e>8 r8 r4 s2
          b2 s2
        }
      >>
    }
  }
>>